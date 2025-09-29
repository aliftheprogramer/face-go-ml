import os
import cv2
import time
import json
import threading
import asyncio
import requests
import numpy as np
import websockets

API_URL = 'http://127.0.0.1:8000/recognize/realtime?min_conf=0.45'
WS_URL = 'ws://127.0.0.1:8000/ws/recognitions'


def main():
    print("Mengirim frame ke API setiap ~0.5-1.0 detik. Tekan 'q' untuk keluar.")

    # Start WebSocket listener thread
    def ws_thread():
        async def listen():
            try:
                async with websockets.connect(WS_URL) as ws:
                    # Send a dummy ping periodically to keep connection
                    async def pinger():
                        while True:
                            try:
                                await ws.send("ping")
                            except Exception:
                                break
                            await asyncio.sleep(15)
                    asyncio.create_task(pinger())
                    while True:
                        msg = await ws.recv()
                        print("WS:", msg)
            except Exception as e:
                print("WS error:", e)
        asyncio.run(listen())

    t = threading.Thread(target=ws_thread, daemon=True)
    t.start()

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    if not cam.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    last_send = 0
    while True:
        ok, frame = cam.read()
        if not ok:
            print("Gagal mengambil frame")
            break

        # draw last results if any
        if hasattr(main, "_last_results"):
            for r in main._last_results:
                t_, rgt, btm, lft = r["box"]
                cv2.rectangle(frame, (lft, t_), (rgt, btm), (0, 255, 255), 2)
                txt = f'{r["label"]} {r["distance"]:.2f}'
                cv2.putText(frame, txt, (lft, max(20, t_-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        now = time.time()
        if now - last_send > 0.7:
            last_send = now
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
                try:
                    r = requests.post(API_URL, files=files, timeout=15)
                    if r.ok:
                        data = r.json()
                        results = data.get("results", [])
                        main._last_results = results
                        # Print dispatch summaries
                        for d in data.get("dispatch", []):
                            label = d.get("label")
                            rep = d.get("report", {})
                            status = rep.get("status")
                            http_status = rep.get("http_status")
                            # When status is "skipped", server returns a "reason" instead of "message"
                            msg = rep.get("message") or rep.get("reason")
                            print(f'dispatch[{label}] -> {status} ({http_status}) {msg}')
                    else:
                        print("HTTP error:", r.status_code, r.text[:200])
                except Exception as e:
                    print("POST error:", e)

        cv2.imshow("Realtime Client (API)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
