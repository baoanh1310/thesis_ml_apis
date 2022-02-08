import os
import cv2
from PIL import Image
from tqdm import tqdm

# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

def load_craft_models():
    # load models
    refine_net = load_refinenet_model(cuda=False)
    craft_net = load_craftnet_model(cuda=False)
    return refine_net, craft_net

def oxygenmeter(refine_net, craft_net):
    # read image
    image = read_image(img_path)



# perform prediction
prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=False,
    long_size=1280
)

# export detected text regions
exported_file_paths = export_detected_regions(
    image=image,
    regions=prediction_result["boxes"],
    output_dir=output_dir,
    rectify=True
)

# export heatmap, detection points, box visualization
export_extra_results(
    image=image,
    regions=prediction_result["boxes"],
    heatmaps=prediction_result["heatmaps"],
    output_dir=output_dir
)

# unload models from gpu
# empty_cuda_cache()

# Perform text recognition with VietOCR
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')

config['weights'] = './vietocr_models/https-COLON--SLASH--SLASH-drive.google.com-SLASH-uc-QUESTION-id-EQUAL-13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False

detector = Predictor(config)

crop_path = os.path.join(output_dir, 'image_crops')
crop_paths = os.listdir(crop_path)
crop_paths = [os.path.join(crop_path, path) for path in crop_paths]

txts = []
for path in crop_paths:
    img = Image.open(path)
    s = detector.predict(img)
    txts.append(s)

print("Text: ", txts)


