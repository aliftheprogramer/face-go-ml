import os
import time
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import face_recognition as fr

EMB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'embeddings')
FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'faces')
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)


class FaceService:
    """
    Core face embedding service using face_recognition.
    - Enroll: save npy embeddings per student_id at data/embeddings/<id>.npy
    - Recognize: return best match with distance for each face in frame
    """
    def __init__(self, tolerance: Optional[float] = None) -> None:
        if tolerance is None:
            try:
                tolerance = float(os.getenv("FACE_TOLERANCE", "0.45"))
            except ValueError:
                tolerance = 0.45
        self.tolerance: float = tolerance
        self._known: Dict[str, np.ndarray] = {}
        self._known_encs: List[np.ndarray] = []
        self._known_labels: List[str] = []
        self._reload_known()

    def _reload_known(self) -> None:
        self._known.clear()
        self._known_encs.clear()
        self._known_labels.clear()
        for f in os.listdir(EMB_DIR):
            if not f.endswith(".npy"):
                continue
            label = os.path.splitext(f)[0]
            arr = np.load(os.path.join(EMB_DIR, f))
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.size == 0:
                continue
            self._known[label] = arr
            for enc in arr:
                self._known_encs.append(enc)
                self._known_labels.append(label)

    @staticmethod
    def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _crop(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        top, right, bottom, left = box
        h, w = img.shape[:2]
        top = max(0, top); left = max(0, left)
        bottom = min(h, bottom); right = min(w, right)
        return img[top:bottom, left:right].copy()

    def enroll_from_image(self, label: str, image_bgr: np.ndarray) -> Dict[str, int]:
        rgb = self._bgr_to_rgb(image_bgr)
        boxes = fr.face_locations(rgb, model="hog")
        if not boxes:
            return {"faces_found": 0, "saved": 0}
        encs = fr.face_encodings(rgb, boxes)
        saved = 0
        save_path = os.path.join(EMB_DIR, f"{label}.npy")
        prev = np.load(save_path) if os.path.exists(save_path) else np.empty((0, 128))
        if prev.ndim == 1 and prev.size == 128:
            prev = prev.reshape(1, -1)
        for box, enc in zip(boxes, encs):
            prev = np.vstack([prev, enc])
            # save cropped face for reference
            crop = self._crop(image_bgr, box)
            dest_dir = os.path.join(FACES_DIR, label)
            os.makedirs(dest_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dest_dir, f"{int(time.time()*1000)}.jpg"), crop)
            saved += 1
        np.save(save_path, prev)
        self._reload_known()
        return {"faces_found": len(boxes), "saved": saved}

    def enroll_from_path(self, label: str, image_path: str) -> Dict[str, int]:
        img = cv2.imread(image_path)
        if img is None:
            return {"faces_found": 0, "saved": 0}
        return self.enroll_from_image(label, img)

    def recognize(self, image_bgr: np.ndarray) -> Dict[str, any]:
        rgb = self._bgr_to_rgb(image_bgr)
        boxes = fr.face_locations(rgb, model="hog")  # returns (top, right, bottom, left)
        encs = fr.face_encodings(rgb, boxes)
        results = []
        for box, enc in zip(boxes, encs):
            label = "Unknown"
            best_dist = 1.0
            if self._known_encs:
                dists = fr.face_distance(self._known_encs, enc)
                idx = int(np.argmin(dists))
                best_dist = float(dists[idx])
                if best_dist <= self.tolerance:
                    label = self._known_labels[idx]
            top, right, bottom, left = map(int, box)
            results.append({
                "box": [top, right, bottom, left],
                "label": label,
                "distance": round(best_dist, 4)
            })
        h, w = image_bgr.shape[:2]
        return {"results": results, "frame_info": {"w": int(w), "h": int(h)}}


if __name__ == "__main__":
    svc = FaceService()
    print(f"Loaded {len(svc.known_encodings)} encodings for {len(set(svc.known_labels))} labels")
