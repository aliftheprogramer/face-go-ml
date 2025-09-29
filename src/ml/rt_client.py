import time
import cv2
import requests
import numpy as np
import threading
import asyncio
import websockets

API_URL = 'http://127.0.0.1:8000/recognize/realtime?min_conf=0.45'


def main():
    # Start WebSocket listener thread
    def ws_thread():
        async def run_ws():
            try:
                async with websockets.connect('ws://127.0.0.1:8000/ws/recognitions') as ws:
                    # send a ping every few seconds to keep alive
                    while True:
                        try:
                            await ws.send('ping')
                            msg = await ws.recv()
                            print(f"WS: {msg}")
                        except Exception:
                            break
                        await asyncio.sleep(1.0)
            except Exception:
                pass
        asyncio.run(run_ws())

    t = threading.Thread(target=ws_thread, daemon=True)
    t.start()
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    if not cam.isOpened():
        print('Error: Tidak dapat membuka kamera')
        return
    print("Mengirim frame ke API setiap ~0.5-1.0 detik. Tekan 'q' untuk keluar.")
    last_sent = 0
    interval = 0.7
    last_results = []
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        now = time.time()
        if now - last_sent >= interval:
            # Encode frame to JPEG
            ok, buf = cv2.imencode('.jpg', frame)
            if ok:
                files = {'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
                try:
                    r = requests.post(API_URL, files=files, timeout=10)
                    if r.ok:
                        data = r.json()
                        last_results = data.get('results', [])
                        dispatch = data.get('dispatch', [])
                        # Print dispatch report(s) if any
                        for d in dispatch:
                            label = d.get('label')
                            report = d.get('report', {})
                            status = report.get('status')
                            code = report.get('http_status')
                            msg = report.get('message')
                            if status in ("sent", "failed"):
                                print(f"dispatch[{label}] -> {status} ({code}) {str(msg)[:120]}")
                except Exception as e:
                    # print once in a while
                    pass
            last_sent = now
        # Draw last results
        for res in last_results:
            (top, right, bottom, left) = res.get('box', [0, 0, 0, 0])
            label = res.get('label', 'Unknown')
            dist = res.get('distance', None)
            txt = label if dist is None else f"{label} ({dist:.2f})"
            color = (0, 255, 0) if label != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, txt, (left, max(0, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow('RT Client (API)', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
