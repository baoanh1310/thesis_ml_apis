import os, time
from tqdm import tqdm
import requests
import glob
import shutil
import argparse
import cv2

from jiwer import wer, cer

FOLDER = os.path.join('..', 'ocr_data', 'FINAL_25_04_TEST')
# FOLDER = os.path.join('..', 'ocr_data', 'FINAL_25_04')
print(FOLDER)

URL = "http://202.191.57.61:5011/predict/"

IMG_FOLDER = './test_imgs_10_06'
if not os.path.exists(IMG_FOLDER):
    os.mkdir(IMG_FOLDER)

# Device

bad = 0
def run(device_name):
	IMG_DEVICE_FOLDER = os.path.join(IMG_FOLDER, device_name)
	if not os.path.exists(IMG_DEVICE_FOLDER):
		os.mkdir(IMG_DEVICE_FOLDER)
	WRONG_DEVICE_FOLDER = os.path.join(IMG_DEVICE_FOLDER, 'wrong')
	if not os.path.exists(WRONG_DEVICE_FOLDER):
		os.mkdir(WRONG_DEVICE_FOLDER)
	TRUE_DEVICE_FOLDER = os.path.join(IMG_DEVICE_FOLDER, 'true')
	if not os.path.exists(TRUE_DEVICE_FOLDER):
		os.mkdir(TRUE_DEVICE_FOLDER)

	print("Device: {}\n".format(device_name))
	DEVICE_FOLDER = os.path.join(FOLDER, device_name)
	GT_FILE = os.path.join(DEVICE_FOLDER, 'gt.txt')

	label_dict = {0: 'ecg_result', 1: 'oxygenmeter_result', 2: 'prescription_result', 3: 'scale_result', 4: 'sphygmomanometer_result', 5: 'thermometer_result'}

	with open(GT_FILE, 'r', encoding='utf8') as f:
		lines = f.readlines()
		lines = [line.strip('\n') for line in lines]

	ground_truth = []
	preds = []
	wrong = 0
	logs = []

	start = time.time()
	for i in tqdm(range(len(lines))):
		line = lines[i]
		img_name, label = line.split('\t')

		# ground_truth.append(str(label))
		img_path = os.path.join(DEVICE_FOLDER, 'images', img_name)
		# img_path = os.path.join(DEVICE_FOLDER, 'easy', img_name)
		# img_path = os.path.join(DEVICE_FOLDER, 'hard', img_name)
		try:
			my_img = {'file': open(img_path, 'rb')}
		except:
			continue
		ground_truth.append(str(label))

		try:
			r = requests.post(URL, files=my_img)
			if r.status_code == 500:
				bad += 1
			cls_num = int(r.json()['data']['result']['classifier_number'])
			key = label_dict[cls_num]
			if cls_num == 0 or cls_num == 2:
				result = str(0.0)
			else:
				if cls_num == 1:
					result = str(r.json()['data']['result']['ocr_result'][key]['oxygen'])
				elif cls_num == 4:
					result = str(r.json()['data']['result']['ocr_result'][key]['blood_pressure'])
				else:
					result = str(r.json()['data']['result']['ocr_result'][key])
			# print("{}, {}".format(cls_num, result))
		except:
			result = str(0.0)

		draw_img = cv2.imread(img_path)
		draw_img = cv2.resize(draw_img, (300, 200), interpolation=cv2.INTER_AREA)
		draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
		cv2.putText(draw_img, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		if result != label:
			wrong += 1
			dst_path = os.path.join(WRONG_DEVICE_FOLDER, img_name)
		else:
			dst_path = os.path.join(TRUE_DEVICE_FOLDER, img_name)

		cv2.imwrite(dst_path, draw_img)

		preds.append(result)
		logs.append("{}\t{}\n".format(label, result))

	end = time.time()
	total = end - start
	print("Total time: {}s".format(total))
	print("Avg time per image: {}".format(total / len(ground_truth)))
	print("Wrong/Total: {}/{}".format(wrong, len(ground_truth)))

	print("WER: {}".format(wer(ground_truth, preds)))
	print("CER: {}\n\n".format(cer(ground_truth, preds)))

	filename = os.path.join('output', '2605_hard_{}.txt'.format(device_name))
	with open(filename, 'w', encoding='utf8') as f:
		for line in logs:
			f.write(line)
		f.write("\n============================================")
		f.write("\nWER: {}\n".format(wer(ground_truth, preds)))
		f.write("CER: {}\n".format(cer(ground_truth, preds)))
		f.write("Avg time per image: {}".format(total / len(ground_truth)))

devices = ['scale', 'thermometer', 'oxygenmeter', 'huyetap']
for device_name in devices:
	run(device_name)

print("Bad: {}".format(bad))
