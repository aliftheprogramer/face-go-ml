import os
from glob import glob
from .face_service import FaceService

KNOWN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'known_faces')

def main():
    svc = FaceService()
    if not os.path.isdir(KNOWN_DIR):
        print(f"Folder tidak ditemukan: {KNOWN_DIR}")
        return

    labels = [d for d in os.listdir(KNOWN_DIR) if os.path.isdir(os.path.join(KNOWN_DIR, d))]
    if not labels:
        print("Tidak ada subfolder di known_faces/. Buat known_faces/<student_id> dan taruh gambar di dalamnya.")
        return

    total = 0
    for label in labels:
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            img_paths.extend(glob(os.path.join(KNOWN_DIR, label, ext)))
        if not img_paths:
            print(f"- {label}: tidak ada gambar, lewati.")
            continue
        saved = 0
        for p in img_paths:
            r = svc.enroll_from_path(label, p)
            saved += r.get("saved", 0)
        total += saved
        print(f"- {label}: {saved} embedding ditambahkan.")
    print(f"Selesai. Total embedding tersimpan: {total}")

if __name__ == '__main__':
    main()
