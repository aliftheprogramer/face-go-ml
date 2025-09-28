import os
import cv2
from .face_service import FaceService

KNOWN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'known_faces')

def main():
    svc = FaceService()
    if not os.path.isdir(KNOWN_DIR):
        print(f"Folder tidak ditemukan: {KNOWN_DIR}")
        return
    labels = [d for d in os.listdir(KNOWN_DIR) if os.path.isdir(os.path.join(KNOWN_DIR, d))]
    total = 0
    for label in labels:
        folder = os.path.join(KNOWN_DIR, label)
        imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in imgs:
            path = os.path.join(folder, img_name)
            bgr = cv2.imread(path)
            if bgr is None:
                print(f"Lewati (tidak bisa dibaca): {path}")
                continue
            n, msg = svc.enroll_from_image(label, bgr)
            print(f"{label}: {img_name} -> {msg}")
            total += n
    print(f"Selesai. Total wajah tersimpan: {total}")

if __name__ == '__main__':
    main()
