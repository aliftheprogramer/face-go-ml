import os
import time
import threading
import requests
from typing import Dict, Any, Optional, Tuple, List


class EventDispatcher:
    """
    Dispatch recognition events to the main Attendance API via a webhook.

        Config via env:
      - ATTENDANCE_WEBHOOK_URL: str (required to enable dispatch)
      - ATTENDANCE_API_KEY: str (optional; sent as header Authorization: Bearer <key>)
            - EVENT_COOLDOWN_SECONDS: int (default 5) minimal interval per student_id
    """

    def __init__(self) -> None:
        self.webhook_url: Optional[str] = os.getenv("ATTENDANCE_WEBHOOK_URL")
        self.api_key: Optional[str] = os.getenv("ATTENDANCE_API_KEY")
        try:
            self.cooldown: int = int(os.getenv("EVENT_COOLDOWN_SECONDS", "5"))
        except ValueError:
            self.cooldown = 5
        self._last_sent: Dict[str, float] = {}
        self._lock = threading.Lock()

    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def _should_send(self, student_id: str) -> Tuple[bool, str]:
        now = time.time()
        with self._lock:
            last = self._last_sent.get(student_id, 0)
            if now - last < self.cooldown:
                return False, f"cooldown_active ({int(self.cooldown - (now - last))}s left)"
            self._last_sent[student_id] = now
        return True, "ok"

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def send(self, payload: Dict[str, Any]) -> Tuple[bool, str, Optional[int]]:
        if not self.enabled():
            return False, "webhook_disabled", None
        try:
            r = requests.post(self.webhook_url, json=payload, headers=self._headers(), timeout=10)
            # Return response body so caller can see tokens/messages from main API
            return r.ok, r.text, r.status_code
        except Exception as e:
            return False, str(e), None

    def maybe_send(self, student_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conditionally send with cooldown. Returns a report dict.
        """
        allowed, reason = self._should_send(student_id)
        if not allowed:
            return {"status": "skipped", "reason": reason}
        ok, msg, code = self.send(payload)
        return {"status": "sent" if ok else "failed", "http_status": code, "message": msg}
