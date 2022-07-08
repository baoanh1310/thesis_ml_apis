import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils import scale_image_size

def oxygenmeter(img_path, refine_net, craft_net, predictor):
    from craft_text_detector import (
        read_image,
        get_prediction,
        export_detected_regions,
        export_extra_results,
        empty_cuda_cache
    )
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

    OUTPUT_DIR = os.path.join('output', 'oxygen')
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # export detected text regions
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        output_dir=OUTPUT_DIR,
        rectify=True
    )

    # export heatmap, detection points, box visualization
    export_extra_results(
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=OUTPUT_DIR
    )


    crop_path = os.path.join(OUTPUT_DIR, 'image_crops')
    crop_paths = os.listdir(crop_path)
    crop_paths = [os.path.join(crop_path, path) for path in crop_paths]

    txts = []
    for path in crop_paths:
        img = Image.open(path)
        s = predictor.predict(img)
        txts.append(s)

    boxes = prediction_result["boxes"]

    def area(box):
        p1, p2, p3, p4 = box[0], box[1], box[2], box[3]
        width = p1[0] - p2[0] if p1[0] > p2[0] else p2[0] - p1[0]
        height = p3[1] - p1[1] if p3[1] > p1[1] else p1[1] - p3[1]
        return width * height

    areas = [area(box) for box in boxes]

    oxygen = 0
    blood_pressure = 0
    bboxes = []
    for i in range(len(areas)):
        try:
            number = int(txts[i])
            bboxes.append((number, areas[i]))
        except:
            continue

    if len(bboxes) == 2:
        oxygen = bboxes[0][0] if bboxes[0][0] > bboxes[1][0] else bboxes[1][0]
        blood_pressure = bboxes[0][0] if bboxes[0][0] < bboxes[1][0] else bboxes[1][0]

        # print("Oxygen: {}%, Blood pressure: {} bpm".format(oxygen, blood_pressure))
    return { "oxygen": oxygen, "blood_pressure": blood_pressure }

def oxygenmeter_new(img_path, predictor, detector):
    results = []
    text_detection = detector.ocr(img_path, rec=False, cls=True)

    # img = cv2.imread(img_path)
    img = scale_image_size(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.open(img_path).convert('RGB')
    # img = np.asarray(img)
    max_area = 0

    oxygenmeter_result = '0'

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
            cv2.imwrite('output_oxy.png', temp)
            oxygenmeter_result = predictor.predict(Image.open('output_oxy.png')) # vietocr rec
        except:
            oxygenmeter_result = '0'

        oxygenmeter_result = oxygenmeter_result.replace('.', '')
        char_arr = [c for c in oxygenmeter_result]
        for i in range(len(char_arr)):
            if char_arr[i] == '[' or char_arr[i] == ']':
                char_arr[i] = '1'

        txt = ''.join(char_arr)
        try:
            txt = float(txt)
            if txt > 100:
                txt = txt / 10
            results.append(txt)
        except:
            results.append(0.0)
    print(results)
    result = max(results) if len(results) > 0 else 0.0
    return result

def oxygenmeter_best(img_path, predictor, detector):
    results = []
    text_detection = detector.ocr(img_path, rec=False, cls=True)

    # img = cv2.imread(img_path)
    img = scale_image_size(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    max_area = 0

    oxygenmeter_result = '0'

    for i, box in enumerate(text_detection):
        top_left = (int(box[0][0]), int(box[0][1]))
        bottom_right = (int(box[2][0]), int(box[2][1]))

        print("Top left: ", top_left)

        # cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)
        try:
            temp = img[top_left[1] - 7: bottom_right[1] + 7, top_left[0]: bottom_right[0]]
        except:
            temp = img[top_left[1] : bottom_right[1], top_left[0]: bottom_right[0]]
        # temp = img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

        height = bottom_right[1] - top_left[1]
        width = bottom_right[0] - top_left[0]
        
        try:
            cv2.imwrite('output_oxy.png', temp)
            oxygenmeter_result = predictor.predict(Image.open('output_oxy.png')) # vietocr rec
        except:
            oxygenmeter_result = '0'

        oxygenmeter_result = oxygenmeter_result.replace('.', '')
        char_arr = [c for c in oxygenmeter_result]
        for i in range(len(char_arr)):
            if char_arr[i] == '[' or char_arr[i] == ']':
                char_arr[i] = '1'

        txt = ''.join(char_arr)
        try:
            txt = float(txt)
            if txt > 100:
                txt = txt / 10
            results.append(txt)
        except:
            results.append(0.0)
    print(results)
    result = max(results) if len(results) > 0 else 0.0
    return result