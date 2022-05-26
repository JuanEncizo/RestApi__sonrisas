"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from unicodedata import name
from PIL import Image
import base64
from io import BytesIO

import torch
from flask import Flask, render_template, request

app = Flask(__name__)

DETECTION_URL = "/detect"

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)
        #results.show()
        data = results.pandas().xyxy[0]#.to_json(orient="split")

        results.imgs # array of original images (as np array) passed to model for inference
        results.render()  # updates results.imgs with boxes and labels

        for img in results.imgs:
          buffered = BytesIO()
          im_base64 = Image.fromarray(img)
          im_base64.save(buffered, format="JPEG")
          #print(base64.b64encode(buffered.getvalue()).decode('utf-8'))  # base64 encoded image with results
          key= "img64"         
          nones = base64.b64encode(buffered.getvalue()).decode('utf-8')

          #data[key]= nones
          #data2 = json.loads(data[0])
          print(data[['name']])
          
          
        return str(data[['name']])+'\n'+ base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/none')
def none():
    return render_template('index.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.conf = 0.7
    #model = torch.hub.load(
       # "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
   # )#.autoshape()  # force_reload = recache latest code
   
    model.eval()
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat

