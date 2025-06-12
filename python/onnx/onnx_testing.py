import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw

# 画像の読み込みと前処理
def preprocess(image_path, input_shape):
    image = Image.open(image_path)
    image_resized = image.resize(input_shape, Image.ANTIALIAS)
    image_np = np.array(image_resized).astype(np.float32)
    image_np = np.transpose(image_np, (2, 0, 1))  # HWC to CHW
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    image_np /= 255.0  # Normalize to [0, 1]
    return image, image_np

# 推論を実行する関数
def run_inference(onnx_model_path, image_path):
    # ONNXランタイムのセッションを作成
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"Failed to load the ONNX model from {onnx_model_path}")
        print(f"Error: {e}")
        return None, None

    # 入力の名前と形状を取得
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape[2:]  # H, W

    # 画像を前処理
    original_image, preprocessed_image = preprocess(image_path, input_shape)

    # 推論を実行
    outputs = ort_session.run(None, {input_name: preprocessed_image})

    return original_image, outputs

# 結果を描画する関数
def draw_boxes(image, boxes, scores, class_ids, class_names, score_threshold=0.3):
    draw = ImageDraw.Draw(image)
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > score_threshold:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1), f'{class_names[class_id]}: {score:.2f}', fill='red')
    return image

# メインの実行部分
if __name__ == "__main__":
    onnx_model_path = "/Users/katti/Desktop/Lab/onnx/best.onnx"
    image_path = "/Users/katti/Desktop/Lab/onnx/image.jpg"
    class_names = ["stairs", "up_stair", "down_stair"]  # クラス名を定義

    original_image, outputs = run_inference(onnx_model_path, image_path)

    if original_image is not None and outputs is not None:
        # 推論結果から必要な情報を抽出（例：ボックス、スコア、クラスID）
        # 以下のコードはYOLOv8の出力形式に依存しますので、適宜修正が必要です
        boxes = outputs[0][:, :4]
        scores = outputs[0][:, 4]
        class_ids = outputs[0][:, 5].astype(int)

        # 結果を描画
        result_image = draw_boxes(original_image, boxes, scores, class_ids, class_names)
        result_image.show()
        result_image.save("/Users/katti/Desktop/Lab/onnx/best.onnx")
