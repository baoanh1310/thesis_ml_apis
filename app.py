import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, flash, json
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable

# VietOCR
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from paddleocr import PaddleOCR

from classifier_network import Resnet18Network
from config import CHECKPOINT
from constant import label_dict

from thermometer import thermometer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VietOCR model
print("Loading VietOCR and PaddleOCR models...")
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained'] = False
config['device'] = device
config['predictor']['beamsearch'] = False

vietocr_predictor = Predictor(config)
paddle_detector = PaddleOCR(lang='en')
print("VietOCR and PaddleOCR model loaded success!")


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = './static/images'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)

# @app.route("/")
# def main():
#     return render_template('index.html')
model_path = './checkpoints/'
class_names = [value for value in label_dict.values()]

# load model
def load_model():
    print("Loading classifier model...")
    num_classes = len(class_names)
    model = Resnet18Network(num_classes)
    model = model.to(device)
    checkpoint = torch.load(CHECKPOINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

# LOAD MODEL
model = load_model()
print("Classifier model loaded success!")

def load_img(img_path):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path)
    image = data_transform(image).float()
    image.unsqueeze_(dim=0)
    image = Variable(image)
    return image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/',methods=['POST'])
def predict():
    if request.method == 'POST':
        print("Request: ", request.files)
        if 'file' not in request.files:
            flash('No key "file"')
            return app.response_class(
                response=json.dumps('No key "file"', cls=NpEncoder),
                status=500,
                mimetype='application/json'
            )
        # read image with "key": "file"
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return app.response_class(
                response=json.dumps('No file selected for uploading'),
                status=500,
                mimetype='application/json'
            )
        # if file and allowed_file(file.filename):
        if file:
            result = dict()

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            img = load_img(filepath)
            img = img.to(device)
            output = model(img)
            _, preds = torch.max(output, 1)
            classifier_result_number = preds.tolist()[0]
            classifier_result = label_dict[classifier_result_number]

            result['classifier_result'] = classifier_result

            if classifier_result_number == 0:
                print("oxygenmeter api")
            elif classifier_result_number == 1:
                print("prescription")
            elif classifier_result_number == 2:
                print("receipt")
            elif classifier_result_number == 3:
                print("scales")
            elif classifier_result_number == 4:
                print("sphygmomanometer")
            elif classifier_result_number == 5:
                ocr_result = thermometer(filepath, vietocr_predictor, paddle_detector)
                result['ocr_result'] = ocr_result

            response = app.response_class(
                response=json.dumps(result, cls=NpEncoder),
                status=200,
                mimetype='application/json'
            )
            return response

        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return app.response_class(
                response=json.dumps('Only allow types txt, pdf, png, jpg, jpeg, gif'),
                status=500,
                mimetype='application/json'
            )

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
            

if __name__ == "__main__":
    # Run app
    app.secret_key = 'ICEBEAR'
    app.run(host="0.0.0.0", port=5123, debug=False)