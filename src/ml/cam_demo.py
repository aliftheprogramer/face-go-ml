import cv2
import time
from .face_service import FaceService


def run():
    svc = FaceService()
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    if not cam.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return
    print("Kamera aktif. Tekan q untuk keluar.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        results = svc.recognize(frame)
        for r in results:
            top, right, bottom, left = r["box"]
            label = f"{r['label']}" if r['distance'] is None else f"{r['label']} ({r['distance']:.2f})"
            color = (0, 255, 0) if r['label'] != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow('Face Attendance', frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
