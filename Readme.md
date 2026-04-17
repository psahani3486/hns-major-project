# Frontend + Backend Setup

This adds a complete local full-stack app for your trained chest X-ray model.

## What Was Added

- Backend API: backend/app.py
- Inference module: backend/inference.py
- Frontend app: frontend/index.html, frontend/styles.css, frontend/app.js

## 1) Install Dependencies

From the project root:

pip install -r requirements.txt


## 2) Run Backend

From the project root:

uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000

Optional: choose a specific deployment bundle:

set MODEL_BUNDLE_PATH=c:\\Users\\Pankaj\\Downloads\\major - Copy\\outputs_paper_seed3_ep10\\resnet18\\lr0.0001_wd0.0001_bs8_ep10\\resnet18_run0_deployment.pt
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000

## 3) Run Frontend

In another terminal:

cd frontend
python -m http.server 5500

Open:

http://127.0.0.1:5500

## API Endpoints

- GET /health
- GET /model
- POST /predict (multipart form field: file)

## Notes

- The backend auto-detects the first \*\_deployment.pt bundle under outputs_paper_seed3_ep10.
- Frontend default backend URL is http://127.0.0.1:8000 and can be changed in the UI.
