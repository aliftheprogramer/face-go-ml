# API Endpoints

Ringkasan endpoint utama untuk server FastAPI di proyek ini. File ini tidak menggantikan README—hanya referensi cepat yang terpisah.

Base URL default: http://localhost:8000

Catatan umum:
- Semua response bertipe JSON kecuali disebutkan lain.
- Untuk upload foto gunakan field name "photo" (multipart/form-data).
- Attendance lokal aktif secara default (ENV ATTENDANCE_LOCAL_ENABLED=true). Jika dimatikan, endpoint attendance tetap tersedia namun tidak ada data baru yang tercatat otomatis.

## Health & Config
- GET /health → { status: "ok" }
- GET /config → konfigurasi environment yang berguna untuk client (mis. webhook, attendance, cooldown)

## Students
- POST /students
  - Form fields:
    - id (string, opsional: jika tidak ada, akan dibuat otomatis UUID v4)
    - full_name (string, wajib)
    - birth_date (string, yyyy-mm-dd, opsional)
    - class_name (string, opsional)
    - address (string, opsional)
    - enroll_after_upload (bool, opsional, default true) → jika true, akan langsung proses enroll embedding dari foto yang diupload
    - photo (file, opsional tapi direkomendasikan)
  - Aksi:
    - Simpan student ke SQLite (data/students.db)
    - Simpan foto ke data/faces/<student_id>/<timestamp>.jpg dan set last_photo_path
    - Jika enroll_after_upload=true dan ada foto: panggil FaceService untuk buat embedding; update embedding_count dan last_enrolled_at
    - Broadcast WS event type "student.enrolled" (jika ada enroll)
  - Contoh response ringkas:
    {
      "id": "abc-123",
      "full_name": "Budi",
      "birth_date": "2010-01-01",
      "class_name": "7A",
      "address": "Jl. Merdeka 1",
      "last_photo_path": "data/faces/abc-123/1759146095327.jpg",
      "embedding_count": 3,
      "last_enrolled_at": "2025-09-28T20:45:00Z",
      "created_at": "2025-09-28T20:40:00Z"
    }

- GET /students
  - Query: q (filter by name contains), limit, offset
  - Response: list of Student

- GET /students/{student_id}
  - Response: Student detail

Catatan: Saat ini PUT/DELETE belum tersedia. Bisa ditambahkan kemudian.

## Enroll & Recognition
- POST /enroll
  - Body: multipart/form-data dengan field image (file) dan opsional student_id
  - Aksi: Proses embedding wajah dari gambar; bila student_id diberikan, asosiakan embedding tersebut. Update embedding_count & last_enrolled_at.
  - Response: hasil dari FaceService (jumlah wajah, dll)

- POST /recognize
  - Body: multipart/form-data dengan field image (file)
  - Aksi: Deteksi dan kenali wajah pada gambar
  - Response: daftar hasil pengenalan dari FaceService

- POST /recognize/realtime
  - Body: multipart/form-data dengan field image (file)
  - Aksi: Sama seperti /recognize, namun juga:
    - Dispatch webhook (jika diaktifkan) dengan cooldown
    - Broadcast WS event ke klien
    - Record attendance lokal untuk wajah yang dikenali (student_id valid)
  - Response: daftar hasil pengenalan (sudah dinormalisasi)

- WebSocket /ws/recognitions
  - Terima event real-time seperti pengenalan atau enroll

## Attendance
- GET /attendance/today
  - Query opsional: class_name, student_id
  - Response: daftar kehadiran hari ini dalam bentuk:
    [
      {
        "student_id": "abc-123",
        "date": "2025-09-28",
        "first_seen_ts": 1695926400.123,
        "last_seen_ts": 1695927400.456,
        "hits": 5
      }
    ]
  - Penjelasan hits: jumlah event pengenalan (recognition events) yang tercatat untuk student tersebut pada tanggal itu. Setiap kali wajah terdeteksi sebagai orang yang sama di /recognize/realtime, hits bertambah 1. Ini bukan jumlah sesi unik—melainkan jumlah total kejadian terdeteksi.

- GET /attendance/student/{student_id}
  - Response: riwayat harian attendance untuk student itu, dengan field yang sama seperti di atas

## Mock
- POST /mock/webhook
  - Endpoint mock untuk menerima webhook lokal (untuk pengujian)

## Tips Integrasi
- CORS sudah diaktifkan. Pastikan origin front-end Anda ada di daftar yang diizinkan bila dibatasi.
- Jika Anda butuh hanya 1 hit per beberapa detik untuk attendance, tambahkan debouncing di sisi klien atau minta kami aktifkan windowing di server.
- ENV penting:
  - ATTENDANCE_LOCAL_ENABLED=true/false
  - ATTENDANCE_LOCAL_DEDUP_SECONDS=60  # 60 detik (demo) atau 0 untuk sekali per hari
  - ATTENDANCE_WEBHOOK_URL, EVENT_COOLDOWN_SECONDS
