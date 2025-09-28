import os
import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Optional, Dict

EMB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'embeddings')
FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'faces')

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)


class FaceService:
    """
    Simple face enrollment and recognition using face_recognition (dlib) encodings.

    - Enroll: extract face encoding(s) from an image; save as .npy with label (student_id).
    - Recognize: compute encoding from frame, compare to known encodings, return best match.
    """

    def __init__(self, tolerance: float = 0.45):
        self.tolerance = tolerance
        self.known_encodings: List[np.ndarray] = []
        self.known_labels: List[str] = []
        self._load_database()

    def _label_to_path(self, label: str) -> str:
        return os.path.join(EMB_DIR, f"{label}.npy")

    def _load_database(self) -> None:
        self.known_encodings.clear()
        self.known_labels.clear()
        for fname in os.listdir(EMB_DIR):
            if not fname.endswith('.npy'):
                continue
            label = os.path.splitext(fname)[0]
            try:
                enc = np.load(os.path.join(EMB_DIR, fname))
                # Handle both single and stacked encodings
                if enc.ndim == 1:
                    self.known_encodings.append(enc)
                    self.known_labels.append(label)
                elif enc.ndim == 2:
                    for i in range(enc.shape[0]):
                        self.known_encodings.append(enc[i])
                        self.known_labels.append(label)
            except Exception:
                continue

    def enroll_from_image(self, label: str, image_bgr: np.ndarray) -> Tuple[int, str]:
        """
        Enroll a label from an image. Returns (num_faces_saved, message)
        """
        if not label:
            return 0, "Label kosong"
        # Convert to RGB for face_recognition
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        if not boxes:
            return 0, "Tidak ada wajah terdeteksi"
        encs = face_recognition.face_encodings(rgb, boxes)
        if not encs:
            return 0, "Gagal mengekstrak embedding"

        # Save individual crop(s) and npy
        face_stack = []
        for i, (top, right, bottom, left) in enumerate(boxes):
            crop = image_bgr[top:bottom, left:right]
            os.makedirs(os.path.join(FACES_DIR, label), exist_ok=True)
            cv2.imwrite(os.path.join(FACES_DIR, label, f"{label}_{i}.jpg"), crop)
            face_stack.append(encs[i])
        enc_array = np.stack(face_stack, axis=0)
        np.save(self._label_to_path(label), enc_array)
        # Reload memory db
        self._load_database()
        return len(face_stack), f"Tersimpan {len(face_stack)} wajah untuk {label}"

    def recognize(self, image_bgr: np.ndarray) -> List[Dict[str, object]]:
        """
        Recognize faces on a BGR frame. Returns list of dict: {box, label, distance}
        box = (top, right, bottom, left)
        """
        if not self.known_encodings:
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        encs = face_recognition.face_encodings(rgb, boxes)
        results = []
        for enc, box in zip(encs, boxes):
            if len(self.known_encodings) == 0:
                results.append({"box": box, "label": "Unknown", "distance": None})
                continue
            dists = face_recognition.face_distance(self.known_encodings, enc)
            if len(dists) == 0:
                results.append({"box": box, "label": "Unknown", "distance": None})
                continue
            idx = int(np.argmin(dists))
            best_dist = float(dists[idx])
            label = self.known_labels[idx] if best_dist <= self.tolerance else "Unknown"
            results.append({"box": box, "label": label, "distance": best_dist})
        return results


if __name__ == "__main__":
    svc = FaceService()
    print(f"Loaded {len(svc.known_encodings)} encodings for {len(set(svc.known_labels))} labels")
