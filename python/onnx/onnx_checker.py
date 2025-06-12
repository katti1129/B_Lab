import onnx

# ONNXモデルの読み込み
model_path = "./train4/weights/best.onnx"
model = onnx.load(model_path)

# モデルのエラーチェック
try:
    onnx.checker.check_model(model)
    print("モデルは有効です。")
except onnx.checker.ValidationError as e:
    print("モデルは無効です: ", e)

# モデルのグラフ構造を表示
model_info = onnx.helper.printable_graph(model.graph)
print(model_info)
