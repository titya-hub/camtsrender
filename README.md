# Cambodia Traffic Signs Detection — Render (no Docker)

This folder deploys the existing Flask interface on Render's **native Python runtime**. It includes the MPWT logo, both model checkpoints, and a `render.yaml` blueprint. No Docker or UI rewrite is required.

## Deploy from GitHub

1. Make a GitHub account if needed, then create a **new public repository** for this deployment. Since this bundle contains only the app and its two model checkpoints, upload the **contents of this folder** to the repository root; do not upload the whole research project or its datasets. Make sure `render.yaml` is at the repository root.
2. Create a free account at [Render](https://dashboard.render.com/register).
3. In the Render dashboard choose **New → Blueprint** and connect the GitHub repository. Render will read `render.yaml`; confirm the service plan is **Free** before creating it.
4. Wait for the first build and deploy. Share the public `onrender.com` address with colleagues.

Colleagues only need a web browser. Images use both models; video and webcam run detection only. Webcam access works in the hosted HTTPS page.

## Free plan and privacy

- The service is public, has no password, and can be used by anyone with the link.
- Uploaded images and video frames are sent to Render for inference. The app does not save the original uploads. Do not upload private or sensitive material.
- Render's free web services sleep after **15 minutes without inbound traffic**; waking the service can take about **one minute**. Free workspaces have **750 web-service instance hours per month**. See [Render's free-service limits](https://render.com/docs/free).
- CPU inference may be slower. Build and try the service with representative images and videos first; if the free instance cannot load the models or respond reliably within its resource limits, a larger paid instance may be necessary.
- The model weights are included in this public repository, so they are public too. Anyone may download them.
- Stop or delete the service in Render when you no longer need it. Free-plan terms and included usage can change; check Render's current pricing and dashboard before deploying.

## Deployment settings

`render.yaml` sets the service root to this folder, installs CPU-only PyTorch dependencies, checks `/api/health`, and starts Flask with Gunicorn bound to Render's `$PORT`. See Render's [Flask deployment guide](https://render.com/docs/deploy-flask) and [Blueprint reference](https://render.com/docs/blueprint-spec).
