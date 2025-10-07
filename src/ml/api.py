#src/ml/api.py

from fastapi import FastAPI, UploadFile, File, Form, Query
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from .face_service import FaceService
from .event_dispatcher import EventDispatcher
import time
import os
from typing import Optional, List
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- DB (SQLite via SQLModel) for Students CRUD ---
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy import text  # for lightweight migrations

DB_PATH = os.getenv("STUDENTS_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "students.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class Student(SQLModel, table=True):
    id: str = Field(primary_key=True, index=True)
    full_name: str
    birth_date: Optional[str] = None  # ISO date string (YYYY-MM-DD)
    class_name: Optional[str] = None
    address: Optional[str] = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    # Optional path of last uploaded photo stored under data/faces/<id>/
    last_photo_path: Optional[str] = None
    # Embedding metadata (derived)
    embedding_count: Optional[int] = 0
    last_enrolled_at: Optional[int] = None


class AttendanceDaily(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str = Field(index=True)
    date: str = Field(index=True, description="YYYY-MM-DD")
    first_seen_ts: int
    last_seen_ts: int
    hits: int = 1

app = FastAPI(title="Face Attendance ML API")
# Allow requests from file:// (Origin null) and other local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local testing; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
svc = FaceService()
dispatcher = EventDispatcher()
ws_clients = set()


@app.on_event("startup")
def on_startup():
    # Create tables lazily
    SQLModel.metadata.create_all(engine)
    # Lightweight migration: ensure 'address' column exists on 'student' table (SQLite only)
    try:
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info('student')"))
            cols = [row[1] for row in res.fetchall()]
            if 'address' not in cols:
                conn.execute(text("ALTER TABLE student ADD COLUMN address TEXT"))
            if 'embedding_count' not in cols:
                conn.execute(text("ALTER TABLE student ADD COLUMN embedding_count INTEGER"))
            if 'last_enrolled_at' not in cols:
                conn.execute(text("ALTER TABLE student ADD COLUMN last_enrolled_at INTEGER"))
            # Ensure unique daily attendance per student per date
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_daily_unique ON attendancedaily(student_id, date)"))
            conn.commit()
    except Exception:
        # best-effort; ignore if not SQLite or already present
        pass

    # Backfill embedding metadata for existing students based on files in data/embeddings
    try:
        emb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'embeddings')
        with Session(engine) as session:
            rows = session.exec(select(Student)).all()
            changed = False
            for s in rows:
                cnt = _embedding_count_for(s.id)
                need_update = False
                if s.embedding_count is None or s.embedding_count != cnt:
                    s.embedding_count = cnt
                    need_update = True
                emb_path = os.path.join(emb_dir, f"{s.id}.npy")
                if (s.last_enrolled_at is None) and os.path.exists(emb_path) and cnt > 0:
                    try:
                        s.last_enrolled_at = int(os.path.getmtime(emb_path))
                        need_update = True
                    except Exception:
                        pass
                if need_update:
                    session.add(s)
                    changed = True
            if changed:
                session.commit()
    except Exception:
        # ignore backfill failure; metadata can be updated later on enroll
        pass


def _embedding_count_for(student_id: str) -> int:
    emb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'embeddings')
    p = os.path.join(emb_dir, f"{student_id}.npy")
    if not os.path.exists(p):
        return 0
    try:
        arr = np.load(p)
        if arr.ndim == 1:
            return 1 if arr.size == 128 else 0
        return int(arr.shape[0])
    except Exception:
        return 0


def _refresh_student_embedding_meta(student_id: str) -> None:
    cnt = _embedding_count_for(student_id)
    with Session(engine) as session:
        st = session.get(Student, student_id)
        if st:
            st.embedding_count = cnt
            st.last_enrolled_at = int(time.time()) if cnt > 0 else st.last_enrolled_at
            session.add(st)
            session.commit()


LOCAL_ATTENDANCE_ENABLED = os.getenv("ATTENDANCE_LOCAL_ENABLED", "true").lower() in ("1","true","yes","on")
try:
    ATTENDANCE_LOCAL_DEDUP_SECONDS = int(os.getenv("ATTENDANCE_LOCAL_DEDUP_SECONDS", "60"))  # demo default: 60s
except ValueError:
    ATTENDANCE_LOCAL_DEDUP_SECONDS = 60


def _record_attendance(student_id: str, ts: Optional[int] = None):
    if not LOCAL_ATTENDANCE_ENABLED:
        return
    ts = ts or int(time.time())
    date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    with Session(engine) as session:
        # Try get today's record
        stmt = select(AttendanceDaily).where(AttendanceDaily.student_id == student_id, AttendanceDaily.date == date_str)
        row = session.exec(stmt).first()
        if row is None:
            row = AttendanceDaily(student_id=student_id, date=date_str, first_seen_ts=ts, last_seen_ts=ts, hits=1)
        else:
            # Dedup policy:
            # - If ATTENDANCE_LOCAL_DEDUP_SECONDS <= 0 => strict once per calendar day
            # - Else => rolling window using last_seen_ts (one hit per window)
            window = ATTENDANCE_LOCAL_DEDUP_SECONDS
            if window <= 0:
                return
            if ts - int(row.last_seen_ts) < max(0, window):
                return
            row.last_seen_ts = ts
            row.hits = int(row.hits or 0) + 1
        session.add(row)
        try:
            session.commit()
        except Exception:
            session.rollback()


def _imdecode_safe(content: bytes):
    """Safely decode bytes -> BGR image or return None if invalid/empty."""
    if not content:
        return None
    npimg = np.frombuffer(content, np.uint8)
    if npimg.size == 0:
        return None
    try:
        bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    except Exception:
        return None
    return bgr


@app.post("/enroll")
async def enroll(student_id: str = Form(...), image: UploadFile = File(...)):
    content = await image.read()
    bgr = _imdecode_safe(content)
    if bgr is None:
        return JSONResponse(status_code=400, content={"ok": False, "msg": "Gambar tidak valid"})
    res = svc.enroll_from_image(student_id, bgr)
    saved = int(res.get("saved", 0))
    if saved > 0:
        _refresh_student_embedding_meta(student_id)
    return {"ok": saved > 0, "saved": saved, "msg": res}


# ---------- Students CRUD (focus: create) ----------
@app.post("/students")
async def create_student(
    student_id: str = Form(..., description="Unique student ID (used as recognition label)"),
    full_name: str = Form(...),
    birth_date: Optional[str] = Form(None, description="YYYY-MM-DD"),
    class_name: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None, description="Optional face image to save & enroll"),
    enroll_after_upload: bool = Form(True, description="If true, immediately enroll the uploaded face")
):
    # Create/Upsert student record (if exists, update metadata only; avoid accidental overwrite)
    with Session(engine) as session:
        existing = session.get(Student, student_id)
        if existing is None:
            st = Student(
                id=student_id,
                full_name=full_name,
                birth_date=birth_date,
                class_name=class_name,
                address=address,
            )
            session.add(st)
            session.commit()
            session.refresh(st)
        else:
            # Update basic fields if provided
            existing.full_name = full_name or existing.full_name
            if birth_date is not None:
                existing.birth_date = birth_date
            if class_name is not None:
                existing.class_name = class_name
            if address is not None:
                existing.address = address
            session.add(existing)
            session.commit()
            session.refresh(existing)
            st = existing

    # If no enroll happens later, try to populate embedding metadata from existing files
    try:
        if photo is None or not enroll_after_upload:
            emb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'embeddings')
            emb_path = os.path.join(emb_dir, f"{student_id}.npy")
            if os.path.exists(emb_path):
                cnt = _embedding_count_for(student_id)
                with Session(engine) as session:
                    s2 = session.get(Student, student_id)
                    if s2:
                        upd = False
                        if s2.embedding_count is None or s2.embedding_count != cnt:
                            s2.embedding_count = cnt
                            upd = True
                        if s2.last_enrolled_at is None:
                            try:
                                s2.last_enrolled_at = int(os.path.getmtime(emb_path))
                                upd = True
                            except Exception:
                                pass
                        if upd:
                            session.add(s2)
                            session.commit()
                        # reflect in response object
                        st.embedding_count = s2.embedding_count
                        st.last_enrolled_at = s2.last_enrolled_at
    except Exception:
        pass

    dispatch: List[dict] = []
    saved = 0
    faces_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'faces', student_id)
    os.makedirs(faces_dir, exist_ok=True)

    if photo is not None:
        content = await photo.read()
        bgr = _imdecode_safe(content)
        if bgr is None:
            return JSONResponse(status_code=400, content={"ok": False, "msg": "Foto tidak valid"})
        # Save original photo to faces dir
        ts = int(time.time() * 1000)
        photo_path = os.path.join(faces_dir, f"{ts}.jpg")
        try:
            ok, buf = cv2.imencode('.jpg', bgr)
            if not ok:
                return JSONResponse(status_code=500, content={"ok": False, "msg": "Gagal menyimpan foto"})
            with open(photo_path, 'wb') as f:
                f.write(buf.tobytes())
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "msg": f"Simpan foto error: {e}"})

        # Update student.last_photo_path
        with Session(engine) as session:
            s2 = session.get(Student, student_id)
            if s2:
                s2.last_photo_path = photo_path
                session.add(s2)
                session.commit()
                # Reflect the change in the current response object
                try:
                    st.last_photo_path = photo_path
                except Exception:
                    pass

        # Optionally enroll immediately
        if enroll_after_upload:
            res = svc.enroll_from_image(student_id, bgr)
            saved = int(res.get("saved", 0))
            # Broadcast WS that a new student was enrolled
            _broadcast_ws({"type": "student.enrolled", "student_id": student_id, "saved": saved})
            # Update embedding meta
            if saved > 0:
                _refresh_student_embedding_meta(student_id)
                # Reload updated meta for response
                with Session(engine) as session:
                    s3 = session.get(Student, student_id)
                    if s3:
                        st.embedding_count = s3.embedding_count
                        st.last_enrolled_at = s3.last_enrolled_at

    return {
        "ok": True,
        "student": {
            "id": st.id,
            "full_name": st.full_name,
            "birth_date": st.birth_date,
            "class_name": st.class_name,
            "address": st.address,
            "last_photo_path": st.last_photo_path,
            "embedding_count": st.embedding_count,
            "last_enrolled_at": st.last_enrolled_at,
        },
        "enrolled": saved,
    }


@app.get("/students")
def list_students(q: Optional[str] = None, limit: int = 50, offset: int = 0):
    with Session(engine) as session:
        stmt = select(Student)
        if q:
            # naive contains filter on id/full_name
            like = f"%{q}%"
            from sqlalchemy import or_  # type: ignore
            stmt = stmt.where(or_(Student.id.contains(q), Student.full_name.contains(q)))
        stmt = stmt.offset(max(0, offset)).limit(max(1, min(200, limit)))
        rows = session.exec(stmt).all()
        return {
            "ok": True,
            "items": [
                {
                    "id": s.id,
                    "full_name": s.full_name,
                    "birth_date": s.birth_date,
                    "class_name": s.class_name,
                    "address": s.address,
                    "last_photo_path": s.last_photo_path,
                    "embedding_count": s.embedding_count,
                    "last_enrolled_at": s.last_enrolled_at,
                }
                for s in rows
            ]
        }


@app.get("/students/{student_id}")
def get_student(student_id: str):
    with Session(engine) as session:
        st = session.get(Student, student_id)
        if not st:
            return JSONResponse(status_code=404, content={"ok": False, "msg": "Student not found"})
        return {
            "ok": True,
            "student": {
                "id": st.id,
                "full_name": st.full_name,
                "birth_date": st.birth_date,
                "class_name": st.class_name,
                "address": st.address,
                "last_photo_path": st.last_photo_path,
                "embedding_count": st.embedding_count,
                "last_enrolled_at": st.last_enrolled_at,
            }
        }


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    content = await image.read()
    bgr = _imdecode_safe(content)
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
    Recognize and immediately dispatch attendance event(s) to the main service, and broadcast via WebSocket.

    Unknown faces are skipped by default (unless send_unknown=true). Cooldown is enforced per student.
    """
    content = await image.read()
    bgr = _imdecode_safe(content)
    if bgr is None:
        return JSONResponse(status_code=400, content={"ok": False, "msg": "Gambar tidak valid"})

    raw = svc.recognize(bgr)
    # Normalize outputs from FaceService: it may return list[dict] or {"results": [...], "frame_info": {...}}
    if isinstance(raw, dict):
        results = raw.get("results", [])
        frame_info = raw.get("frame_info", {})
        w = int(frame_info.get("w", bgr.shape[1]))
        h = int(frame_info.get("h", bgr.shape[0]))
    else:
        results = raw
        h, w = bgr.shape[:2]

    sent_reports = []
    now = int(time.time())
    tol = svc.tolerance
    for r in results:
        # Guard against malformed element
        if not isinstance(r, dict):
            continue
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
            # Send webhook without blocking the event loop (prevents deadlock when calling same server)
            report = await run_in_threadpool(dispatcher.maybe_send, label, payload)
            sent_reports.append({"label": label, "report": report})
            # Broadcast to WS subscribers
            _broadcast_ws({
                "type": "recognized",
                "student_id": label,
                "distance": dist,
                "ts": now,
                "dispatch": report
            })
            # Record local daily attendance only for known faces
            if is_known:
                _record_attendance(label, now)
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
        "attendance_local_enabled": LOCAL_ATTENDANCE_ENABLED,
        "attendance_local_dedup_seconds": ATTENDANCE_LOCAL_DEDUP_SECONDS,
    }


# --- Attendance query endpoints ---
@app.get("/attendance/today")
def attendance_today(limit: int = 100, offset: int = 0):
    today = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
    with Session(engine) as session:
        stmt = select(AttendanceDaily).where(AttendanceDaily.date == today).offset(max(0, offset)).limit(max(1, min(500, limit)))
        rows = session.exec(stmt).all()
        # Map with student names
        out = []
        for a in rows:
            st = session.get(Student, a.student_id)
            out.append({
                "student_id": a.student_id,
                "full_name": st.full_name if st else None,
                "date": a.date,
                "first_seen_ts": a.first_seen_ts,
                "last_seen_ts": a.last_seen_ts,
                "hits": a.hits,
            })
        return {"ok": True, "items": out}


@app.get("/attendance/student/{student_id}")
def attendance_for_student(student_id: str, days: int = 7):
    with Session(engine) as session:
        stmt = select(AttendanceDaily).where(AttendanceDaily.student_id == student_id).order_by(AttendanceDaily.date.desc()).limit(max(1, min(60, days)))
        rows = session.exec(stmt).all()
        return {
            "ok": True,
            "student_id": student_id,
            "items": [
                {
                    "date": r.date,
                    "first_seen_ts": r.first_seen_ts,
                    "last_seen_ts": r.last_seen_ts,
                    "hits": r.hits,
                } for r in rows
            ]
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


# (deduplicated above)


# --- Mock webhook for testing tokens locally (optional) ---
@app.post("/mock/webhook")
async def mock_webhook(payload: dict):
    # pretend we issue a token for attendance; echo back
    token = f"ATT-{int(time.time())}-{payload.get('student_id','UNKNOWN')}"
    return {"ok": True, "token": token, "received": payload}
