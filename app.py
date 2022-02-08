import os
from flask import Flask, render_template, request, flash, json
from werkzeug.utils import secure_filename
import requests

from config import *
from utils import *
from thermometer import thermometer

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = './static/images'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)

# LOAD MODELS
classifier_model = load_classifier_model()
vietocr_predictor = load_vietocr_model()
paddle_detector = load_paddleocr_model()


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
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)
            
            classifier_result_number, classifier_result = get_classifier_results(img_path)

            result['classifier_result'] = classifier_result
            result['classifier_number'] = classifier_result_number

            if classifier_result_number == 0:
                print("oxygenmeter api")

            elif classifier_result_number == 1:
                try:
                    dictToSend = {'file': file}
                    res = requests.post(PRESCRIPTION_API, json=dictToSend)
                    ocr_result = res.json()
                except:
                    ocr_result = "prescription"
                result['ocr_result'] = ocr_result

            elif classifier_result_number == 2:
                try:
                    dictToSend = {'file': file}
                    res = requests.post(RECEIPT_API, json=dictToSend)
                    ocr_result = res.json()
                except:
                    ocr_result = "receipt"
                result['ocr_result'] = ocr_result

            elif classifier_result_number == 3:
                print("scales")

            elif classifier_result_number == 4:
                print("sphygmomanometer")

            elif classifier_result_number == 5:
                ocr_result = thermometer(img_path, vietocr_predictor, paddle_detector)
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