# Face Attendance ML (Microservice-first)

This project provides a minimal ML microservice to enroll face embeddings and recognize faces for student attendance.

## Features
- Enroll: POST an image with `student_id` to store embeddings
- Recognize: POST an image and get predicted labels and distances
- Webcam demo to visualize in real-time

## Install
Create a venv (optional) and install requirements.

## Run API
- Start server: `uvicorn src.ml.api:app --reload`
- POST /enroll (multipart/form-data): fields `student_id`, `image`
- POST /recognize (multipart/form-data): field `image`
 - POST /recognize/realtime (multipart/form-data): field `image` + query `min_conf` (optional) `send_unknown` (optional)
 - GET /health, GET /config
	- WebSocket: `ws://<host>/ws/recognitions` (server pushes events when recognized)

## Data Layout
- `data/embeddings/<student_id>.npy`: saved embeddings
- `data/faces/<student_id>/*.jpg`: cropped face images for reference

## Webcam Demo
`python -m src.ml.cam_demo`

## Notes
- Uses `face_recognition` (dlib) for encodings; CPU HOG model by default.
- Adjust tolerance in `FaceService(tolerance=0.45)` for stricter/looser matching.

## Environment (.env)
Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Supported variables:
- `ATTENDANCE_WEBHOOK_URL` — main API endpoint to receive events (leave empty to disable)
- `ATTENDANCE_API_KEY` — optional Bearer token for the webhook
- `EVENT_COOLDOWN_SECONDS` — default 60 seconds per student
- `FACE_TOLERANCE` — override face match tolerance (default 0.45)

## Realtime flow (Microservice → Main Attendance API)

This ML service can push events to your main Attendance API via webhook when a face is recognized.

Set env vars before starting the server:

```bash
export ATTENDANCE_WEBHOOK_URL="https://your-main-api.example.com/attendance/events"
export ATTENDANCE_API_KEY="<optional-key>"
export EVENT_COOLDOWN_SECONDS=60   # per-student cooldown to avoid duplicates
uvicorn src.ml.api:app --reload
```

Call realtime endpoint from the client:

```bash
curl -X POST "http://127.0.0.1:8000/recognize/realtime?min_conf=0.45&send_unknown=false" \
	-F image=@sample.jpg
```

Dispatch event payload (JSON sent to your webhook):

```json
{
	"event": "attendance.recognized",
	"student_id": "siswaA",
	"distance": 0.38,
	"ts": 1727580000,
	"frame_info": {"w": 640, "h": 480},
	"box": {"top": 120, "right": 260, "bottom": 220, "left": 160}
}
```

Response from ML realtime endpoint:

```json
{
	"ok": true,
	"results": [
		{"box": [120,260,220,160], "label": "siswaA", "distance": 0.38}
	],
	"dispatch": [
		{"label": "siswaA", "report": {"status": "sent", "http_status": 200, "message": "sent"}}
	],
	"webhook_enabled": true
}
```

Main Attendance API expected to respond 2xx. If `EVENT_COOLDOWN_SECONDS` is set, duplicate events for the same student within the window are skipped.


## Full ML API reference

- POST `/enroll`
	- Request (multipart/form-data):
		- `student_id`: string (required)
		- `image`: file (required; JPG/PNG)
	- Response 200:
		- `{ ok: boolean, saved: number, msg: string }`

- POST `/recognize`
	- Request (multipart/form-data):
		- `image`: file
	- Response 200:
		- `{ ok: true, results: [ { box: [top,right,bottom,left], label: string, distance: number|null } ] }`

- POST `/recognize/realtime`
	- Request (multipart/form-data + query):
		- `image`: file
		- `min_conf`: float optional (default uses `FACE_TOLERANCE`/service tolerance)
		- `send_unknown`: bool optional (default false)
	- Behavior: recognize faces, dispatch event(s) to webhook if known (and passes threshold), then return immediate result
	- Response 200:
		- `{ ok: true, results: [...], dispatch: [ { label: string, report: { status: 'sent'|'failed'|'skipped', http_status?: number, message?: string } } ], webhook_enabled: boolean }`

- GET `/health`
	- Response 200: `{ ok: true, encodings: number, labels: number }`

- GET `/config`
	- Response 200: `{ tolerance: number, webhook_enabled: boolean, cooldown_seconds: number }`

- WebSocket `ws://<host>/ws/recognitions`
	- Server push message example:
	```json
	{
		"type": "recognized",
		"student_id": "siswaA",
		"distance": 0.37,
		"ts": 1727580000,
		"dispatch": { "status": "sent", "http_status": 200, "message": "{\"ok\":true,\"token\":\"ATT-...\"}" }
	}
	```

- Mock webhook (testing only): POST `/mock/webhook`
	- Responds `{ ok: true, token: "ATT-<ts>-<student_id>", received: <payload> }`


## How to build your Attendance Web API (integration guide)

Your Attendance Web API should expose a webhook endpoint to receive recognition events from this ML service.

1) Create a webhook endpoint
	- Method: `POST`
	- Suggested path: `/attendance/events`
	- Auth: expect `Authorization: Bearer <ATTENDANCE_API_KEY>` (optional but recommended)
	- Request body (JSON) from ML:
		```json
		{
			"event": "attendance.recognized",
			"student_id": "siswaA",
			"distance": 0.38,
			"ts": 1727580000,
			"frame_info": {"w": 640, "h": 480},
			"box": {"top": 120, "right": 260, "bottom": 220, "left": 160}
		}
		```

2) Validate and deduplicate
	- Validate presence of `event`, `student_id`, `ts`
	- Optional: enforce a max age (e.g., reject if `now - ts > 30s`)
	- Deduplicate events per `student_id` using a cooldown window to avoid double attendance:
		- If last attendance for `student_id` < N seconds ago (e.g., 60s), ignore or return 200 with a note

3) Persist attendance
	- Insert record: `{ student_id, timestamp, source: 'ml', device_id?, class_id?, distance }`
	- Return a JSON response with your reference token/ID:
		```json
		{ "ok": true, "token": "ATT-2025-09-29-XYZ", "student_id": "siswaA" }
		```
	- The ML service will surface your response body back to clients (HTTP response text is returned in `dispatch.message`)

4) Security recommendations
	- Require `ATTENDANCE_API_KEY` and verify Bearer token
	- Optionally restrict IP allow-list
	- Add rate-limits to prevent abuse

5) Optional fields and extensions
	- You may want ML to include more context (set these in your own design and update ML accordingly):
		- `device_id`, `location_id`, `class_id`, `session_id`
		- A signed HMAC header (e.g., `X-Signature: HMAC_SHA256(body, SHARED_SECRET)`) to verify integrity

6) Local testing without your API
	- Set `ATTENDANCE_WEBHOOK_URL=http://127.0.0.1:8000/mock/webhook`
	- Start ML server, run realtime client. You’ll see tokens echoed in terminal logs and WS messages.


## Example client flows

- Simple: post still images
```bash
curl -X POST "http://127.0.0.1:8000/recognize/realtime?min_conf=0.45&send_unknown=false" \
	-F image=@known_faces/siswaA/img1.jpg
```

- Realtime webcam client (HTTP + WebSocket)
```bash
python -m src.ml.rt_client
```
Logs will show lines like:
```
dispatch[siswaA] -> sent (200) {"ok":true,"token":"ATT-..."}
WS: {"type":"recognized","student_id":"siswaA",...}
```


source venv/bin/activate