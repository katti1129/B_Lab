import coremltools as ct

# ONNXモデルのパス
onnx_model_path = "/Users/katti/Desktop/Lab/test/onnx/best.onnx"

# CoreMLモデルに変換
coreml_model = ct.converters.onnx.convert(model=onnx_model_path)

# CoreMLモデルを保存
save_path = "/Users/katti/Desktop/Lab/YOLOv8.mlmodel"
coreml_model.save(save_path)
