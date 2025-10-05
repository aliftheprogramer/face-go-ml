# Face Attendance ML (Layanan ML untuk Absensi Wajah)

Proyek ini adalah layanan microservice berbasis FastAPI untuk mendaftarkan (enroll) embedding wajah dan mengenali (recognize) wajah dari gambar atau aliran kamera. Tujuan utama: mudah dijalankan di CPU, data lokal sederhana, dan integrasi gampang ke API utama via webhook + WebSocket.

## Fitur
- Enroll gambar: simpan embedding 128-D per orang ke `data/embeddings/<id>.npy`
- Recognize: prediksi label (ID) dan jarak (distance) per wajah pada gambar
- Mode realtime:
  - Dispatch event ke API utama melalui webhook (opsional)
  - Broadcast notifikasi ke klien via WebSocket
- Demo lokal: webcam (`src/ml/cam_demo.py`) dan klien realtime (`src/ml/rt_client.py`)

## Persyaratan Sistem (Linux)
- Python 3.10+ (contoh berjalan di 3.12)
- Kamera (opsional) untuk demo webcam
- Paket sistem umum untuk OpenCV/dlib:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libgl1 libglib2.0-0
```

Catatan: `face_recognition` memakai dlib; wheel pra-bangun biasanya tersedia. Jika butuh kompilasi dari sumber, pastikan toolchain di atas siap.

## Instalasi
Jalankan di folder project.

```bash
# 1) Buat dan aktifkan virtualenv
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# 2) Pasang dependensi Python
pip install -r requirements.txt
```

Jika pemasangan `face_recognition`/OpenCV gagal, pastikan paket sistem pada bagian Persyaratan sudah terpasang, lalu ulangi `pip install -r requirements.txt`.

## Menjalankan Server API
Pilih salah satu:

```bash
# Opsi A: jalankan dari modul
uvicorn src.ml.api:app --reload

# Opsi B: jalankan via file main.py
uvicorn main:app --reload
```

Cek cepat:
- Swagger Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Jika muncul `ModuleNotFoundError: fastapi`, pastikan venv aktif dan `pip install -r requirements.txt` sudah dilakukan.

## Cara Training / Enroll Model
“Training” = mengumpulkan embedding 128-D untuk tiap orang yang akan dikenali. Embedding disimpan per ID ke file `.npy`.

Ada 2 cara:

1) Enroll satu gambar via REST API
```bash
curl -X POST "http://127.0.0.1:8000/enroll" \
  -F student_id=alip \
  -F image=@known_faces/alip/IMG_20230927_203408.jpg
```
Respon contoh:
```json
{ "ok": true, "saved": 1, "msg": {"faces_found": 1, "saved": 1} }
```

2) Enroll massal (bulk) dari folder `known_faces/`
Struktur contoh:
```
known_faces/
  alip/
    foto1.jpg
    foto2.jpg
  yoga/
    a.jpg
    b.jpg
```
Jalankan:
```bash
python -m src.ml.bulk_enroll
```
Hasil embedding disimpan di `data/embeddings/<label>.npy`. Cropped face juga disimpan di `data/faces/<label>/*.jpg` sebagai referensi.

Tips data:
- Sertakan beberapa foto dengan variasi pose/pencahayaan untuk tiap orang
- Pastikan wajah jelas (frontal/near-frontal) dan resolusi memadai

## Cara Kerja Face Recognition (Ringkas)
1) Konversi BGR (OpenCV) → RGB
2) Deteksi wajah dengan HOG (`face_recognition.face_locations(..., model="hog")`)
3) Ekstraksi embedding 128-D (`face_recognition.face_encodings`)
4) Bandingkan embedding baru ke semua embedding dikenal (Euclidean distance)
5) Ambil jarak minimum. Jika `distance <= tolerance` → cocok; jika tidak → "Unknown"

Parameter penting:
- `tolerance` default 0.45 (lebih kecil = lebih ketat)
  - Dapat diatur via env `FACE_TOLERANCE` atau parameter `min_conf` pada endpoint realtime

Output per wajah:
- `box`: [top, right, bottom, left]
- `label`: ID yang dikenali atau `Unknown`
- `distance`: jarak terbaik (semakin kecil → semakin mirip)

## Referensi Endpoint
Semua endpoint berjalan di host/port yang Anda jalankan (default `127.0.0.1:8000`).

### POST `/enroll`
- Form-data:
  - `student_id`: string (wajib)
  - `image`: file (wajib)
- Respon 200:
```json
{ "ok": true, "saved": 1, "msg": {"faces_found": 1, "saved": 1} }
```

### POST `/recognize`
- Form-data: `image` (wajib)
- Respon 200:
```json
{ "ok": true, "results": [ {"box":[120,260,220,160], "label":"alip", "distance":0.38} ] }
```

### POST `/recognize/realtime`
- Query opsional:
  - `min_conf`: float (default 0.0). Jika 0 → gunakan `tolerance` server.
  - `send_unknown`: bool (default false). Jika true → Unknown juga diproses/dikirim event.
- Form-data: `image` (wajib)
- Perilaku:
  - Mengembalikan hasil seperti `/recognize`
  - Jika cocok atau `send_unknown=true`, server:
    - Mengirim event ke webhook (jika diaktifkan via env)
    - Broadcast WS ke `/ws/recognitions`
- Respon 200 (contoh):
```json
{
  "ok": true,
  "results": [ {"box":[120,260,220,160], "label":"alip", "distance":0.38} ],
  "dispatch": [ { "label": "alip", "report": {"status":"sent","http_status":200,"message":"..."} } ],
  "webhook_enabled": true
}
```

### GET `/health`
```json
{ "ok": true, "encodings": 25, "labels": 2 }
```

### GET `/config`
```json
{ "tolerance": 0.45, "webhook_enabled": true, "cooldown_seconds": 60 }
```

### WebSocket `/ws/recognitions`
- Contoh pesan dari server:
```json
{ "type":"recognized", "student_id":"alip", "distance":0.38, "ts":1727580000, "dispatch": {"status":"sent"} }
```

### POST `/mock/webhook`
- Untuk tes lokal: mengembalikan token dummy dan echo payload.

## Realtime & Webhook (Integrasi dengan API Utama)
Aktifkan dengan environment variable:
```bash
export ATTENDANCE_WEBHOOK_URL="https://api-utama.example.com/attendance/events"
export ATTENDANCE_API_KEY="<opsional-token>"
export EVENT_COOLDOWN_SECONDS=5   # jeda minimal per orang (default 5)
```
Jika `ATTENDANCE_WEBHOOK_URL` kosong → webhook nonaktif (tetap ada WS broadcast dan respon lokal).

Ringkas alur integrasi:
- ML memanggil webhook API utama dengan payload:
```json
{
  "event": "attendance.recognized",
  "student_id": "alip",
  "distance": 0.38,
  "ts": 1727580000,
  "frame_info": {"w": 640, "h": 480},
  "box": {"top":120,"right":260,"bottom":220,"left":160}
}
```
- API utama memvalidasi, deduplikasi, dan menyimpan absensi, lalu merespons 2xx.
- Respons API utama akan diteruskan kembali sebagai `dispatch[].report.message` di respon endpoint realtime dan dikirim ke WS.

## Struktur Folder
```
project/
  data/
    embeddings/            # .npy per student_id (embedding 128-D)
    faces/
      <student_id>/        # crop wajah yang disimpan saat enroll
  known_faces/             # sumber gambar untuk bulk_enroll (input)
  src/ml/
    api.py                 # FastAPI endpoints
    face_service.py        # logika core: enroll & recognize
    event_dispatcher.py    # kirim event ke API utama via webhook
    cam_demo.py            # demo webcam lokal
    rt_client.py           # klien realtime (HTTP + WS)
  main.py                  # entry opsional untuk uvicorn main:app
```

## Demo & Contoh Pakai
- Demo Webcam (lokal):
```bash
python -m src.ml.cam_demo
```
- Klien Realtime (HTTP + WS):
```bash
python -m src.ml.rt_client
```
- Contoh cepat (copy-paste):
```bash
# Jalankan server
source venv/bin/activate
uvicorn src.ml.api:app --reload

# Enroll satu foto
curl -X POST "http://127.0.0.1:8000/enroll" \
  -F student_id=alip \
  -F image=@known_faces/alip/IMG_20230927_203408.jpg

# Recognize sekali
curl -X POST "http://127.0.0.1:8000/recognize" \
  -F image=@known_faces/alip/IMG_20230927_203408.jpg

# Recognize realtime + kirim event (opsional) + WS broadcast
export ATTENDANCE_WEBHOOK_URL="http://127.0.0.1:8000/mock/webhook"
curl -X POST "http://127.0.0.1:8000/recognize/realtime?min_conf=0.45&send_unknown=false" \
  -F image=@known_faces/alip/IMG_20230927_203408.jpg
```

## Troubleshooting (Masalah Umum)
- `ModuleNotFoundError: fastapi` atau paket lain
  - Aktifkan venv: `source venv/bin/activate`
  - Pasang deps: `pip install -r requirements.txt`
  - Jalankan uvicorn dengan Python venv: `python -m uvicorn src.ml.api:app --reload`
- Gagal pasang `dlib`/`face_recognition`
  - Pastikan paket sistem: `build-essential cmake libgl1 libglib2.0-0`
  - Upgrade alat: `pip install -U pip setuptools wheel`
- Kamera tidak terbuka (demo webcam)
  - Coba index kamera lain: `cv2.VideoCapture(1)` atau `2`
  - Pastikan tidak sedang dipakai aplikasi lain
- Tidak ada wajah terdeteksi saat enroll/recognize
  - Coba gambar lain yang lebih jelas; posisi frontal/near-frontal
- WebSocket tidak menerima pesan
  - Pastikan endpoint `/recognize/realtime` dipanggil dan server aktif
  - Periksa firewall/port jika akses dari perangkat lain
- Event webhook selalu `skipped`
  - Mekanisme cooldown aktif. Atur `EVENT_COOLDOWN_SECONDS` atau tunggu jeda.

## Catatan Keamanan
- Endpoint publik tidak memiliki autentikasi bawaan. Jika dipublikasikan, gunakan API Gateway/Reverse Proxy dengan autentikasi & rate-limit.
- Webhook mendukung Bearer token via `ATTENDANCE_API_KEY`.

Selamat mencoba! Untuk kustomisasi pipeline (mis. ganti ke CNN atau menambah metadata), mulai dari `src/ml/face_service.py` dan endpoint di `src/ml/api.py`.
