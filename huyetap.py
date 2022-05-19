import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

def huyetap(img_path, predictor, detector):
    results = []
    text_detection = detector.ocr(img_path, rec=False, cls=True)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.open(img_path).convert('RGB')
    # img = np.asarray(img)
    max_area = 0

    huyetap_result = '0'

    for i, box in enumerate(text_detection):
        top_left = (int(box[0][0]), int(box[0][1]))
        bottom_right = (int(box[2][0]), int(box[2][1]))

        # cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)
        try:
            temp = img[top_left[1] - 7: bottom_right[1] + 7, top_left[0]: bottom_right[0]]
        except:
            temp = img[top_left[1] : bottom_right[1], top_left[0]: bottom_right[0]]
        # temp = img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

        height = bottom_right[1] - top_left[1]
        width = bottom_right[0] - top_left[0]
        
        try:
            cv2.imwrite('output_huyetap.png', temp)
            huyetap_result = predictor.predict(Image.open('output_huyetap.png')) # vietocr rec
        except:
            huyetap_result = '0'

        huyetap_result = huyetap_result.replace('.', '')
        char_arr = [c for c in huyetap_result]
        for i in range(len(char_arr)):
            if char_arr[i] == '[' or char_arr[i] == ']':
                char_arr[i] = '1'

        txt = ''.join(char_arr)
        try:
            txt = float(txt)
            if txt > 140:
                txt = txt / 10
            results.append(txt)
        except:
            results.append(0.0)
    print(results)
    result = max(results) if len(results) > 0 else 0.0
    return result