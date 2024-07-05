from ultralytics import YOLO
from flask import Flask, request, Response, render_template
from PIL import Image
import json

app = Flask(__name__)

model = YOLO("../modèles/yolov8m.pt") #choose de model you want to apply on your website

@app.route("/")
def home():
    return render_template('./index.html')

@app.route("/detect", methods=["POST"])
def detect():
    try:
        buf = request.files["image_file"]
        image = Image.open(buf.stream)
        boxes = detect_objects_on_image(image)
        return Response(
            json.dumps(boxes),
            mimetype='application/json'
        )
    except Exception as e:
        return Response(
            json.dumps({"error": str(e)}),
            mimetype='application/json',
            status=500
        )

def detect_objects_on_image(image):
    results = model(image)
    result = results[0]
    filtered_boxes = [box for box in result.boxes if box.conf[0].item() >= 0.5]
    output = []

    for box in filtered_boxes.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
