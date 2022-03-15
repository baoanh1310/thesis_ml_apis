import os
import cv2
from PIL import Image
from tqdm import tqdm

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
