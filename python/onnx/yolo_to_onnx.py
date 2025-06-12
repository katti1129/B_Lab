##########OK#################onnxファイルインポートできる################
# CupSoup_Detection_ONNX.py
#

from ultralytics import YOLO

model_path = "/Users/katti/Desktop/Lab/best.pt"
onnx_path = "/Users/katti/Desktop/Lab/best.onnx"

# convert to ONNX model
model = YOLO(model_path)
model.export(format='onnx', imgsz=[640,640])