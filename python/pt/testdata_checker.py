from ultralytics import YOLO
# 学習済みモデルをロード
model = YOLO("/Users/katti/Desktop/best.pt")
# テストデータで評価
metrics = model.val(data="/Users/katti/Desktop/final.v5i.yolov8/data.yaml",split="test")
# 結果を表示
print(metrics)