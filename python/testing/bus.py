from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to CoreML format
model.export(format="coreml")  # creates 'yolov8n.mlpackage'

# Load the exported CoreML model
coreml_model = YOLO("yolov8n.mlpackage")

# Run inference
results = coreml_model("https://ultralytics.com/images/bus.jpg")