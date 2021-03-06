import os
from flask import Flask, render_template, request, flash, json
from werkzeug.utils import secure_filename
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import *
from utils import *
from thermometer import thermometer, thermometer_new, thermometer_moi
from oxygenmeter import oxygenmeter, oxygenmeter_new, oxygenmeter_best
from huyetap import huyetap, huyetap_best
from scale import scale, scale_new
from ecg import ecg

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = './static/images'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
# limiter = Limiter(
#     app,
#     key_func=get_remote_address,
#     default_limits=["200 per day", "50 per hour"]
# )

# @app.errorhandler(429)
# def ratelimit_handler(e):
#     failed_res = {
#         'appStatus': -1,
#         'data': {}
#     }
#     return make_response(
#                 failed_res,
#                 429,
#             )

# LOAD MODELS
classifier_model = load_classifier_model()
vietocr_predictor = load_vietocr_model()
paddle_detector = load_paddleocr_model()
# load CRAFT models
# refine_net, craft_net = load_craft_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
# @limiter.limit("1/minute")
def default():
    return "Icebear"

@app.route('/predict/',methods=['POST'])
def predict():
    res = dict()
    res['appStatus'] = 0
    result = { 'ocr_result': {} }
    res['data'] = { 
        'result': result
    }
    failed_res = {
        'appStatus': -1,
        'data': {}
    }
    if request.method == 'POST':
        print("Request: ", request.files)
        if 'file' not in request.files:
            flash('No key "file"')
            # res['appStatus'] = -1
            return app.response_class(
                #response=json.dumps('No key "file"', cls=NpEncoder),
                response=json.dumps(failed_res, cls=NpEncoder),
                status=500,
                mimetype='application/json'
            )
        # read image with "key": "file"
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            # res['appStatus'] = -1
            return app.response_class(
                # response=json.dumps('No file selected for uploading'),
                response=json.dumps(failed_res, cls=NpEncoder),
		status=500,
                mimetype='application/json'
            )
        # if file and allowed_file(file.filename):
        if file:
            # result = dict()

            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)
            
            classifier_result_number, classifier_result = get_classifier_results(img_path, classifier_model)

            # result['classifier_result'] = classifier_result
            result['classifier_number'] = classifier_result_number
            # result['ocr_result'] = 0.0

            # ECG
            if classifier_result_number == 0:
                print("ECG")
                result['ocr_result']['ecg_result'] = ecg(img_path) # return array shape (12x504 elements)
                print(len(result['ocr_result']['ecg_result']))
            # Oxygenmeter
            elif classifier_result_number == 1:
                # ocr_result = oxygenmeter(img_path, refine_net, craft_net, vietocr_predictor)
                # ocr_result = { "oxygen": 0.0, "blood_pressure": 0.0 }
                # ocr_result = { "oxygen": 0.0, "heart_rate": 0.0 }
                # ocr_result["oxygen"] = oxygenmeter_new(img_path, vietocr_predictor, paddle_detector)
                # ocr_result["oxygen"] = oxygenmeter_best(img_path, vietocr_predictor, paddle_detector)
                # ocr_result = oxygenmeter_new(img_path, vietocr_predictor, paddle_detector)
                ocr_result = oxygenmeter_best(img_path, vietocr_predictor, paddle_detector)
                print("Oxygenmeter: ", ocr_result)
                
                result['ocr_result']['oxygenmeter_result'] = ocr_result

            # Prescription
            elif classifier_result_number == 2:
                try:
                    dictToSend = {'file': file}
                    res = requests.post(PRESCRIPTION_API, json=dictToSend)
                    ocr_result = res.json()
                except:
                    ocr_result = {
                        "name": "",
                        "drugs": [
                            {
                                "price": 0,
                                "quantity": 0,
                                "drug_name": ""
                            }
                        ]
                    }
                result['ocr_result']['medicine_receipt_result'] = ocr_result

            # Scale
            elif classifier_result_number == 3:
                # print("scales")
                # ocr_result = scale(img_path, vietocr_predictor, paddle_detector)
                ocr_result = scale_new(img_path, vietocr_predictor, paddle_detector)
                result['ocr_result']['scale_result'] = ocr_result

            # Sphygmomanometer
            elif classifier_result_number == 4:
                # temporary value
                # ocr_result = {
                #     "blood_pressure": huyetap_best(img_path, vietocr_predictor, paddle_detector)
                # }
                ocr_result = huyetap_best(img_path, vietocr_predictor, paddle_detector)
                print("Sphygmomanometer: ", ocr_result)
                # ocr_result = huyetap(img_path, vietocr_predictor, paddle_detector)
                result['ocr_result']['sphygmomanometer_result'] = ocr_result

            # Thermometer
            elif classifier_result_number == 5:
                # ocr_result = thermometer(img_path, vietocr_predictor, paddle_detector)
                # ocr_result = thermometer_new(img_path, vietocr_predictor, paddle_detector)
                ocr_result = thermometer_moi(img_path, vietocr_predictor, paddle_detector)
                result['ocr_result']['thermometer_result'] = ocr_result
                
            # Unknown
            elif classifier_result_number == 100:
                ocr_result = ""
                result['ocr_result']['unknown_result'] = ocr_result

            res['data']['result'] = result
            response = app.response_class(
                response=json.dumps(res, cls=NpEncoder),
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
    app.run(host="0.0.0.0", port=5011, debug=False)
