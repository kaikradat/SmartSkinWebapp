from pathlib import Path
import os
import uuid

import cv2
from flask import Flask, render_template, request
from PIL import Image
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.environ.get("UPLOAD_FOLDER", APP_DIR / "static" / "uploads"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", APP_DIR / "best.pt"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
model = None


def get_model():
    """Load YOLO only when image analysis is requested, not on every web startup."""
    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))
    return model


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #แบบประเมินความเสี่ยง
        answers = [request.form.get(f'q{i}') for i in range(1, 10)]
        risk_score = answers.count("yes")
        is_risk = risk_score >= 3

        #รับภาพ
        uploaded_file = request.files.get("skin_image")
        if uploaded_file:
            img = Image.open(uploaded_file.stream).convert("RGB")
            img_filename = f"{uuid.uuid4().hex}_original.jpg"
            img_path = UPLOAD_DIR / img_filename
            img.save(img_path)

            #วิเคราะห์ภาพ
            detector = get_model()
            results = detector.predict(str(img_path), conf=0.25)
            result = results[0]

            # detected img + แปลงสี
            detect_filename = img_filename.replace("original", "detected")
            detect_path = UPLOAD_DIR / detect_filename
            plotted = result.plot()
            rgb_image = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_image).save(detect_path)

            #แยกชื่อโรคและ%ความมั่นใจ
            diseases = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = detector.names[class_id]
                diseases.append((class_name, round(conf * 100, 2)))

            if not diseases:
                diseases.append(('ปกติ', 0))

            original_rel_path = f"uploads/{img_filename}"
            detect_rel_path = f"uploads/{detect_filename}"

            return render_template("index.html",
                                   submitted=True,
                                   risk_score=risk_score,
                                   is_risk=is_risk,
                                   original_image=original_rel_path,
                                   detection_image=detect_rel_path,
                                   diseases=diseases)

    return render_template("index.html", submitted=False)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
