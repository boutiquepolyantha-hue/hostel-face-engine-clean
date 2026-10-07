# Hostel face engine prototype

This separate FastAPI service provides `/health`, `/enroll`, and `/verify` for
the hostel backend. It uses OpenCV LBPH only as a controlled prototype. It is
not a production-grade biometric system: add liveness detection, encrypted
templates, consent, retention/deletion controls, and a persistent disk or
database before real deployment.

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Set the same `FACE_ENGINE_TOKEN` on this service and the hostel backend. The
service expects the token in `X-Face-Engine-Token`.

## Render settings

Create a separate Render **Web Service** from this repository:

```text
Build command: pip install -r requirements.txt
Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

After deployment, copy the service URL and set this on the hostel backend:

```text
FACE_ENGINE_URL=https://your-face-engine.onrender.com/verify
FACE_ENGINE_TOKEN=<the same secret used here>
```

## Enroll a guardian

After the guardian is approved and has a guardian code, send one or more
consented face images to `/enroll` using the same token. The image can be a
`data:image/jpeg;base64,...` value captured by the website camera.

```json
{
  "guardian_code": "G-00001",
  "face_images": ["data:image/jpeg;base64,..."]
}
```
