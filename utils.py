import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image

from constant import label_dict, CLASSIFIER_THRESHOLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vietocr_model():
    from PIL import Image
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    from config import VIETOCR_CHECKPOINT

    # Load VietOCR model
    print("Loading VietOCR model...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['vocab'] = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ '
    config['weights'] = VIETOCR_CHECKPOINT
#    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained'] = False
    config['device'] = device
    config['predictor']['beamsearch'] = False
    
    vietocr_predictor = Predictor(config)
    print("VietOCR model loaded success!")

    return vietocr_predictor

def load_paddleocr_model():
    from paddleocr import PaddleOCR
    print("Loading PaddleOCR model...")
    paddle_detector = PaddleOCR(lang='en')
    print("PaddleOCR model loaded success!")

    return paddle_detector

def load_classifier_model():
    from classifier_network import Resnet18Network, SqueezeNet, MobileNetV3Small, Resnet34Network
    from config import CLASSIFIER_CHECKPOINT, SQUEEZENET_CHECKPOINT, MOBILENET_CHECKPOINT, RESNET34_CHECKPOINT

    print("Loading classifier model...")
    class_names = [value for value in label_dict.values()]
    num_classes = len(class_names)
    model = Resnet34Network(num_classes)
    # model = Resnet18Network(num_classes)
    # model = SqueezeNet(num_classes)
    model = model.to(device)
    checkpoint = torch.load(RESNET34_CHECKPOINT)
    # checkpoint = torch.load(CLASSIFIER_CHECKPOINT)
    # checkpoint = torch.load(SQUEEZENET_CHECKPOINT, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Classifier model loaded success!")

    return model

def _preprocess_classifier_image(img_path):
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

def get_classifier_results(img_path, classifier_model):
    # classifier_model = load_classifier_model()
    img = _preprocess_classifier_image(img_path)
    img = img.to(device)

    output = classifier_model(img)
    values, preds = torch.max(output, 1)
    value = values.tolist()[0]
    if value < CLASSIFIER_THRESHOLD:
        return 100, 'unknown'
    classifier_result_number = preds.tolist()[0]
    classifier_result = label_dict[classifier_result_number]

    return classifier_result_number, classifier_result

def load_craft_models():
    print("Loading CRAFT models...")
    # import craft functions
    from craft_text_detector import (
        load_craftnet_model,
        load_refinenet_model,
    )
    # load models
    refine_net = load_refinenet_model(cuda=False)
    craft_net = load_craftnet_model(cuda=False)
    print("CRAFT models loaded success!")

    return refine_net, craft_net
