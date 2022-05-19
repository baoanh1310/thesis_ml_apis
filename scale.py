import os

import cv2
import numpy as np
from PIL import Image

def scale(img_path, predictor, detector):
    img = cv2.imread(img_path)

    area = img.shape[0] * img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    edges = cv2.Canny(blur, 30, 150, apertureSize=3)

    start_y = 0

    for i, point in enumerate(edges):
        if len(point[point == 255]) > 20:
            start_y = i
            break

    img_temp = img[start_y: start_y +
                   int(img.shape[0] / 5), :]
    # cv2.imwrite('output_can.png', img_temp)

    gray_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    blur_temp = cv2.blur(gray_temp, (3, 3))
    edges_temp = cv2.Canny(blur_temp, 30, 150, apertureSize=3)

    # text_detection = detector.ocr(img_temp, cls=False, rec=False)

    contours, hierarchy = cv2.findContours(edges_temp,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    temp = []
    result = 0

    for i in contours:
        if cv2.contourArea(i) < 0.005 * area:
            continue
        rect = cv2.boundingRect(i)
        x, y, w, h = rect
        cropped = img_temp[y: y+h, x: x+w]
        temp.append(i)
        # cv2.imwrite('test.png', cropped)

        text_detection = detector.ocr(img_temp, rec=False)

        # print(text_detection)

        for j in text_detection:
            # print
            try:
                # result = int(j[-1][0].replace('.', ''))
                result = float(j[-1][0])
                break
            except:
                continue

    return result

def scale_new(img_path, predictor, detector):
    results = []
    text_detection = detector.ocr(img_path, rec=False, cls=True)

    img = cv2.imread(img_path)
    scale_result = '0'

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
            cv2.imwrite('output_can.png', temp)
            scale_result = predictor.predict(Image.open('output_can.png')) # vietocr rec
        except:
            scale_result = '0'

    scale_result = scale_result.replace('.', '')
    char_arr = [c for c in scale_result]
    for i in range(len(char_arr)):
        if char_arr[i] == '[' or char_arr[i] == ']':
            char_arr[i] = '1'

    txt = ''.join(char_arr)
    print("Raw result: ", txt)
    try:
        txt = float(txt)
        if txt >= 100:
            txt = txt / 10
        results.append(str(txt))
    except:
        results.append('')
    result = results[0]
    try:
        result = float(result)
    except:
        result = 0.0
    return result