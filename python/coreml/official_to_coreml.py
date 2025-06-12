from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # load an official model
model.export(format='coreml', nms=True)
