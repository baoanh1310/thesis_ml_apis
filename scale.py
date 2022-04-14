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

    text_detection = detector.ocr(img_temp)

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

        text_detection = detector.ocr(cropped)

        # print(text_detection)

        for j in text_detection:
            # print
            try:
                result = int(j[-1][0].replace('.', ''))
                break
            except:
                continue

    return result