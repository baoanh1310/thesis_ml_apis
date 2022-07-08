from scale import scale_new
from thermometer import thermometer_moi
from oxygenmeter import oxygenmeter_best
from utils import load_vietocr_model, load_paddleocr_model

paddle = load_paddleocr_model()
vietocr = load_vietocr_model()

# scale_new('./weight.jpg', vietocr, paddle)
# print(thermometer_moi('./sample/ther3.jpg', vietocr, paddle))
print(oxygenmeter_best('./sample/oxy4.jpg', vietocr, paddle))
