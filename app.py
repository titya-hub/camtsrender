import base64
import binascii
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DETECTOR_PATH = Path(
    os.environ.get(
        "CAMTS_DETECTOR_MODEL",
        str(BASE_DIR / "runs" / "detect" / "train5" / "weights" / "best.pt"),
    )
)
CLASSIFIER_PATH = Path(
    os.environ.get(
        "CAMTS_CLASSIFIER_MODEL",
        str(BASE_DIR / "runs" / "classify" / "tsr_model_v1" / "weights" / "best.pt"),
    )
)
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
BRAND_GREEN = (63, 157, 18)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 1024 * 1024
app.config["MAX_IMAGE_PIXELS"] = MAX_IMAGE_PIXELS
app.logger.setLevel(logging.INFO)

_detector = None
_classifier = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


class ModelUnavailableError(RuntimeError):
    pass


def _load_model(path, model_type):
    if not path.is_file():
        raise ModelUnavailableError(f"The {model_type} model was not found: {path}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ModelUnavailableError(
            "Ultralytics is not installed. Install the project dependencies first."
        ) from error

    try:
        return YOLO(str(path))
    except (OSError, RuntimeError, ValueError) as error:
        raise ModelUnavailableError(
            f"Could not load the {model_type} model: {error}"
        ) from error


def get_detector():
    global _detector
    if _detector is None:
        with _model_lock:
            if _detector is None:
                _detector = _load_model(DETECTOR_PATH, "detection")
    return _detector


def get_classifier():
    global _classifier
    if _classifier is None:
        with _model_lock:
            if _classifier is None:
                _classifier = _load_model(CLASSIFIER_PATH, "recognition")
    return _classifier


def decode_image(image_bytes):
    if not image_bytes:
        raise ValueError("Choose a non-empty image.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Images must be 16 MB or smaller.")

    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("The selected file is not a supported image.")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Images must be 25 megapixels or smaller.")
    return image


def decode_data_url(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("A valid camera frame is required.")

    header, encoded_image = data_url.split(",", 1)
    if header not in (
        "data:image/jpeg;base64",
        "data:image/png;base64",
        "data:image/webp;base64",
    ):
        raise ValueError("Camera frames must be JPEG, PNG, or WebP images.")

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("The camera frame could not be decoded.") from error

    return decode_image(image_bytes)


def detect(image, recognize=False):
    height, width = image.shape[:2]
    detector = get_detector()
    detector_result = detector.predict(source=image, verbose=False)[0]
    detections = []

    if detector_result.boxes is None:
        return detections

    boxes = detector_result.boxes.xyxy.cpu().tolist()
    class_ids = detector_result.boxes.cls.cpu().tolist()
    confidences = detector_result.boxes.conf.cpu().tolist()
    classifier = get_classifier() if recognize else None

    for coordinates, class_id, confidence in zip(boxes, class_ids, confidences):
        x1 = max(0, min(width, int(coordinates[0])))
        y1 = max(0, min(height, int(coordinates[1])))
        x2 = max(0, min(width, int(coordinates[2])))
        y2 = max(0, min(height, int(coordinates[3])))
        if x2 <= x1 or y2 <= y1:
            continue

        detection_class = detector.names.get(int(class_id), str(int(class_id)))
        detection = {
            "box": [x1, y1, x2, y2],
            "detection_class": str(detection_class),
            "detection_confidence": float(confidence),
            "class_name": str(detection_class),
        }

        if classifier is not None:
            crop = image[y1:y2, x1:x2]
            classification_result = classifier.predict(
                source=crop, verbose=False
            )[0]
            probabilities = classification_result.probs
            if probabilities is not None:
                class_index = int(probabilities.top1)
                detection["class_name"] = str(
                    classification_result.names.get(class_index, class_index)
                )
                detection["recognition_confidence"] = float(
                    probabilities.top1conf.item()
                )

        detections.append(detection)

    return detections


def annotate_image(image, detections):
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BRAND_GREEN, 3)
        label = f'{detection["class_name"]}  {detection["detection_confidence"]:.0%}'
        if "recognition_confidence" in detection:
            label += f'  |  {detection["recognition_confidence"]:.0%}'

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(image.shape[1] / 1600, 0.8))
        (label_width, label_height), baseline = cv2.getTextSize(
            label, font, font_scale, 2
        )
        label_top = max(0, y1 - label_height - baseline - 10)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(image.shape[1], x1 + label_width + 12), y1),
            BRAND_GREEN,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 6, max(label_height, y1 - baseline - 5)),
            font,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "detector_model": DETECTOR_PATH.is_file(),
            "classifier_model": CLASSIFIER_PATH.is_file(),
            "ready": DETECTOR_PATH.is_file() and CLASSIFIER_PATH.is_file(),
        }
    )


@app.post("/api/detect-image")
def detect_image():
    upload = request.files.get("image")
    if upload is None:
        return jsonify(error="Choose an image to analyze."), 400

    try:
        image = decode_image(upload.read(MAX_IMAGE_BYTES + 1))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    try:
        with _inference_lock:
            detections = detect(image, recognize=True)
    except ModelUnavailableError as error:
        app.logger.error("%s", error)
        return jsonify(error=str(error)), 503
    except (OSError, RuntimeError, ValueError) as error:
        app.logger.exception("Image inference failed")
        return jsonify(error=f"Image inference failed: {error}"), 500

    success, encoded_image = cv2.imencode(
        ".jpg", annotate_image(image, detections), [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not success:
        app.logger.error("OpenCV could not encode the annotated image")
        return jsonify(error="The annotated image could not be generated."), 500

    image_url = "data:image/jpeg;base64," + base64.b64encode(
        encoded_image.tobytes()
    ).decode("ascii")
    return jsonify(detections=detections, image=image_url)


@app.post("/api/detect-frame")
def detect_frame():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="A camera or video frame is required."), 400

    try:
        image = decode_data_url(payload.get("image"))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    try:
        with _inference_lock:
            detections = detect(image)
    except ModelUnavailableError as error:
        app.logger.error("%s", error)
        return jsonify(error=str(error)), 503
    except (OSError, RuntimeError, ValueError) as error:
        app.logger.exception("Frame inference failed")
        return jsonify(error=f"Frame inference failed: {error}"), 500

    return jsonify(
        detections=detections,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="The upload is too large. Choose a smaller image or frame."), 413


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
    )
