#Load Flask
from flask import Flask, request, Response
from imageai.Detection import VideoObjectDetection
from waitress import serve
import os
from PIL import Image
from ultralytics import YOLO
import json
from util import * 

app = Flask(__name__)

@app.route("/")
def root():

    with open("index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image"]
    print(buf)
    
    img = detect_objects_on_image(Image.open(buf.stream))
    return Response(
        json.dumps(img),
        mimetype='application/json'
    )

def detect_objects_in_video(buf):
    vid_obj_detect = VideoObjectDetection()
    vid_obj_detect.setModelPath("weights/best3.pt")
    vid_obj_detect.loadModel()
def detect_objects_on_image(buf):
    model = YOLO("weights/best3.pt")
    img = buf.resize((640, 640))
    results = model.predict(img)
    result = results[0]
    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, model.names[class_id], prob
        ])
    return output

if __name__=="__main__":
    print(app)
    app.run(host='0.0.0.0')
    serve(app, host='0.0.0.0', port=5000)

