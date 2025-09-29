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
    print("Kamera aktif. Tekan 'q' untuk keluar.")
    last = 0
    while True:
        ok, frame = cam.read()
        if not ok:
            print("Gagal mengambil frame")
            break
        # limit inference fps ~10
        now = time.time()
        if now - last > 0.08:
            out = svc.recognize(frame)
            last = now
        else:
            out = getattr(run, "_last_out", {"results": []})
        results = out["results"]
        for r in results:
            t, rgt, b, lft = r["box"]
            cv2.rectangle(frame, (lft, t), (rgt, b), (0, 255, 0), 2)
            txt = f'{r["label"]} {r["distance"]:.2f}'
            cv2.putText(frame, txt, (lft, max(20, t - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        run._last_out = out
        cv2.imshow("Face Recognition (local)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
