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

## Data Layout
- `data/embeddings/<student_id>.npy`: saved embeddings
- `data/faces/<student_id>/*.jpg`: cropped face images for reference

## Webcam Demo
`python -m src.ml.cam_demo`

## Notes
- Uses `face_recognition` (dlib) for encodings; CPU HOG model by default.
- Adjust tolerance in `FaceService(tolerance=0.45)` for stricter/looser matching.
