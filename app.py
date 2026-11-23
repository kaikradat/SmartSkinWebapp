from flask import Flask, render_template, request
from PIL import Image
from ultralytics import YOLO
import os
import uuid
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


model = YOLO("best.pt")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            img.save(img_path)

            #วิเคราะห์ภาพ
            results = model.predict(img_path, conf=0.25)
            result = results[0]

            # detected img + แปลงสี
            detect_filename = img_filename.replace("original", "detected")
            detect_path = os.path.join(app.config['UPLOAD_FOLDER'], detect_filename)
            plotted = result.plot()
            rgb_image = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_image).save(detect_path)

            #แยกชื่อโรคและ%ความมั่นใจ
            diseases = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[class_id]
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
    app.run(host="localhost", port=5000, debug=True)
