from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from .face_service import FaceService

app = FastAPI(title="Face Attendance ML API")
svc = FaceService()


@app.post("/enroll")
async def enroll(student_id: str = Form(...), image: UploadFile = File(...)):
    content = await image.read()
    npimg = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"ok": False, "msg": "Gambar tidak valid"})
    n, msg = svc.enroll_from_image(student_id, bgr)
    return {"ok": n > 0, "saved": n, "msg": msg}


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    content = await image.read()
    npimg = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"ok": False, "msg": "Gambar tidak valid"})
    results = svc.recognize(bgr)
    return {"ok": True, "results": results}
