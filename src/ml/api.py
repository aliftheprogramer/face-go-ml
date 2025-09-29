from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
import numpy as np
import cv2
from .face_service import FaceService
from .event_dispatcher import EventDispatcher
import time
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = FastAPI(title="Face Attendance ML API")
svc = FaceService()
dispatcher = EventDispatcher()
ws_clients = set()


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


@app.post("/recognize/realtime")
async def recognize_realtime(
    image: UploadFile = File(...),
    min_conf: float = Query(0.0, description="Max allowed distance to consider recognized; <= tolerance used."),
    send_unknown: bool = Query(False, description="If true, also send events for Unknown faces."),
):
    """
    Recognize and immediately dispatch attendance event(s) to the main service.

    Dispatch contract (JSON):
      {
        "event": "attendance.recognized",
        "student_id": "<label>",
        "distance": <float>,
        "ts": <unix_seconds>,
        "frame_info": {"w": <int>, "h": <int>},
        "box": {"top": int, "right": int, "bottom": int, "left": int}
      }
    Unknown faces are skipped by default (unless send_unknown=true with student_id="Unknown").
    Cooldown is enforced per student via EventDispatcher.
    """
    content = await image.read()
    npimg = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"ok": False, "msg": "Gambar tidak valid"})
    h, w = bgr.shape[:2]
    results = svc.recognize(bgr)
    sent_reports = []
    now = int(time.time())
    tol = svc.tolerance
    for r in results:
        label = r.get("label", "Unknown")
        dist = r.get("distance", None)
        (top, right, bottom, left) = r.get("box", (0, 0, 0, 0))
        # Filter by recognition quality
        is_known = label != "Unknown" and dist is not None and dist <= max(0.0, min_conf or tol)
        if is_known or (send_unknown and label == "Unknown"):
            payload = {
                "event": "attendance.recognized",
                "student_id": label,
                "distance": dist,
                "ts": now,
                "frame_info": {"w": w, "h": h},
                "box": {"top": top, "right": right, "bottom": bottom, "left": left},
            }
            report = dispatcher.maybe_send(label, payload)
            sent_reports.append({"label": label, "report": report})
    return {"ok": True, "results": results, "dispatch": sent_reports, "webhook_enabled": dispatcher.enabled()}


@app.get("/health")
async def health():
    return {"ok": True, "encodings": len(svc.known_encodings), "labels": len(set(svc.known_labels))}


@app.get("/config")
async def config():
    return {
        "tolerance": svc.tolerance,
        "webhook_enabled": dispatcher.enabled(),
        "cooldown_seconds": int(os.getenv("EVENT_COOLDOWN_SECONDS", "60")),
    }


# --- WebSocket: push recognition updates to subscribers ---
@app.websocket("/ws/recognitions")
async def ws_recognitions(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            # Keep connection alive; client may send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


def _broadcast_ws(message: dict):
    import json
    data = json.dumps(message)
    dead = []
    for ws in list(ws_clients):
        try:
            # send in background best-effort (FastAPI doesn't provide built-in task here)
            import anyio
            anyio.from_thread.run(ws.send_text, data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.discard(ws)


# Modify realtime endpoint to also push to WS
@app.post("/recognize/realtime")
async def recognize_realtime(
    image: UploadFile = File(...),
    min_conf: float = Query(0.0, description="Max allowed distance to consider recognized; <= tolerance used."),
    send_unknown: bool = Query(False, description="If true, also send events for Unknown faces."),
):
    # ... existing code kept ...
    content = await image.read()
    npimg = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"ok": False, "msg": "Gambar tidak valid"})
    h, w = bgr.shape[:2]
    results = svc.recognize(bgr)
    sent_reports = []
    now = int(time.time())
    tol = svc.tolerance
    for r in results:
        label = r.get("label", "Unknown")
        dist = r.get("distance", None)
        (top, right, bottom, left) = r.get("box", (0, 0, 0, 0))
        is_known = label != "Unknown" and dist is not None and dist <= max(0.0, min_conf or tol)
        if is_known or (send_unknown and label == "Unknown"):
            payload = {
                "event": "attendance.recognized",
                "student_id": label,
                "distance": dist,
                "ts": now,
                "frame_info": {"w": w, "h": h},
                "box": {"top": top, "right": right, "bottom": bottom, "left": left},
            }
            report = dispatcher.maybe_send(label, payload)
            sent_reports.append({"label": label, "report": report})
            # Broadcast to WS subscribers
            _broadcast_ws({
                "type": "recognized",
                "student_id": label,
                "distance": dist,
                "ts": now,
                "dispatch": report
            })
    return {"ok": True, "results": results, "dispatch": sent_reports, "webhook_enabled": dispatcher.enabled()}


# --- Mock webhook for testing tokens locally (optional) ---
@app.post("/mock/webhook")
async def mock_webhook(payload: dict):
    # pretend we issue a token for attendance; echo back
    token = f"ATT-{int(time.time())}-{payload.get('student_id','UNKNOWN')}"
    return {"ok": True, "token": token, "received": payload}
