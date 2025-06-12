from ultralytics import YOLO

# カスタムモデルのロード
model = YOLO("/Users/katti/Desktop/best.pt")  # 例: 'best.pt'ファイルのパス

# Core ML形式にエクスポート
model.export(format='coreml', nms=True)
