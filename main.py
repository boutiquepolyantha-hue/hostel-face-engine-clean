"""Small prototype face-verification API for the hostel project.

This uses OpenCV LBPH and a Haar cascade. It is suitable for a controlled
prototype only; production should use an embedding model, liveness detection,
encrypted template storage, and a persistent database/disk.
"""

import base64
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Hostel Face Engine", version="0.1.0")
DATA_DIR = Path(os.getenv("FACE_DATA_DIR", "data"))
MODEL_FILE = DATA_DIR / "lbph.yml"
GALLERY_FILE = DATA_DIR / "gallery.json"
LOCK = threading.Lock()
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class EnrollRequest(BaseModel):
    guardian_code: str = Field(min_length=3, max_length=30)
    face_images: list[str] = Field(min_length=1, max_length=8)


class VerifyRequest(BaseModel):
    source: str = "website"
    face_image: str = Field(min_length=100, max_length=4_000_000)


def check_token(token: str | None) -> None:
    expected = os.getenv("FACE_ENGINE_TOKEN", "").strip()
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Invalid face-engine token")


def decode_image(value: str) -> np.ndarray:
    try:
        encoded = value.split(",", 1)[1] if "," in value else value
        image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception as reason:
        raise HTTPException(status_code=422, detail="Invalid face image") from reason
    if image is None:
        raise HTTPException(status_code=422, detail="Invalid face image")
    return image


def crop_face(image: np.ndarray) -> np.ndarray:
    faces = CASCADE.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No clear face found in image")
    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    return cv2.resize(image[y:y + height, x:x + width], (160, 160))


def load_gallery() -> dict[str, int]:
    if not GALLERY_FILE.exists():
        return {}
    return json.loads(GALLERY_FILE.read_text(encoding="utf-8"))


def rebuild_model() -> None:
    gallery = load_gallery()
    images, labels = [], []
    for guardian_code, label in gallery.items():
        for path in DATA_DIR.glob(f"{label}_*.png"):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(label)
    if not images:
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_FILE))


@app.get("/health")
def health():
    return {"status": "ok", "enrolled_guardians": len(load_gallery())}


@app.post("/enroll")
def enroll(payload: EnrollRequest, x_face_engine_token: str | None = Header(default=None)):
    check_token(x_face_engine_token)
    code = payload.guardian_code.strip()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK:
        gallery = load_gallery()
        label = gallery.get(code)
        if label is None:
            label = max(gallery.values(), default=0) + 1
            gallery[code] = label
        for index, value in enumerate(payload.face_images):
            face = crop_face(decode_image(value))
            cv2.imwrite(str(DATA_DIR / f"{label}_{index}.png"), face)
        GALLERY_FILE.write_text(json.dumps(gallery, indent=2), encoding="utf-8")
        rebuild_model()
    return {"enrolled": True, "guardian_code": code, "images": len(payload.face_images)}


@app.post("/verify")
def verify(payload: VerifyRequest, x_face_engine_token: str | None = Header(default=None)):
    check_token(x_face_engine_token)
    if not MODEL_FILE.exists():
        raise HTTPException(status_code=503, detail="No guardian faces have been enrolled")
    with LOCK:
        image = crop_face(decode_image(payload.face_image))
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(MODEL_FILE))
        label, confidence = recognizer.predict(image)
        gallery = load_gallery()
        reverse = {value: code for code, value in gallery.items()}
        guardian_code = reverse.get(label)
    # LBPH confidence is distance-like: lower means a closer match.
    threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "75"))
    verified = guardian_code is not None and confidence <= threshold
    if not verified:
        return {"verified": False, "guardian_code": "", "identity_match": 0}
    return {
        "verified": True,
        "guardian_code": guardian_code,
        "identity_match": max(0, min(100, round(100 - confidence))),
    }
