import os

import cv2
import numpy as np
from PIL import Image

def thermometer(img_path, predictor, detector):
    # Read image
    text_detection = detector.ocr(img_path, rec=False)

    img = cv2.imread(img_path)
    max_area = 0

    OUTPUT_DIR = os.path.join('output', 'nhietke')
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    OUTPUT_PATH = os.path.join(OUTPUT_DIR, '{}.png'.format(len(os.listdir(OUTPUT_DIR))))

    for i, box in enumerate(text_detection):
        top_left     = (int(box[0][0]), int(box[0][1]))
        bottom_right = (int(box[2][0]), int(box[2][1]))
    
        # cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)
        temp = img[top_left[1] - 7: bottom_right[1] + 7, top_left[0]: bottom_right[0]]

        height = bottom_right[1] - top_left[1]
        width = bottom_right[0] - top_left[0]

        if height * width > max_area:
            max_area = height * width
            cv2.imwrite(OUTPUT_PATH, cv2.resize(temp, (300, 300)))

    def preprocess(img_path):
        img = cv2.imread(img_path, 0)

        ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

        return Image.fromarray(thresh1)

    result = ''.join(predictor.predict(preprocess(OUTPUT_PATH)).split())

    char_arr = [c for c in result]
    for i in range(len(char_arr)):
        if char_arr[i] == '[' or char_arr[i] == ']':
            char_arr[i] = '1'
    
    result = ''.join(char_arr)
    try:
        result = float(result)
        if result >= 100:
            result = result / 10
    except:
        print("Cannot casting to numeric value")

    # delete output image
    # os.remove(OUTPUT_PATH)
    
    return result