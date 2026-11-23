# SmartSkin

Flask and YOLO demo for skin image analysis and a risk questionnaire.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`. The app reads `best.pt` from the repository directory by default. Set `MODEL_PATH`, `UPLOAD_FOLDER`, or `PORT` to override those paths and settings.

## Deploy a public web app

GitHub Pages can host static files, but it cannot run this Flask server or the YOLO model. This repository includes `render.yaml` for a Render web service. Connect the repository in Render and deploy the Blueprint; Render will build the Python app and provide a public `onrender.com` URL.

The free Render service can sleep when idle, so its first request after inactivity may take longer. Its local filesystem is temporary; uploaded images and generated results can disappear when the service restarts or deploys.

## Privacy and intended use

Uploaded skin images are processed by the server and saved under `static/uploads` while the service is running. Do not upload identifiable or sensitive photos to a public demo. This project is not a medical device and its output is not a diagnosis; consult a qualified clinician for medical concerns.

Uploaded images and generated results are excluded from Git by `.gitignore`.
