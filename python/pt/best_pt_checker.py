#############OK###########真正面からの推論では強い##################
from ultralytics import YOLO
import os

# トレーニング済みモデルの読み込み
model = YOLO("/Users/katti/Desktop/best.pt")

# 検出を行う画像のパス
image_path = "/Users/katti/Desktop/IMG_1465.MOV"

desktop_path = os.path.expanduser("~/Desktop/yolo_results")

# 階段を検出
results = model.predict(source=image_path, imgsz=640, save=True, project=desktop_path)

# 検出結果の表示
#for result in results:
    #result.show()


"""
import cv2
from ultralytics import YOLO

# トレーニング済みモデルの読み込み
model = YOLO("/Users/katti/Desktop/Lab/best.pt")

# 検出を行う画像のパス
image_path = "/Users/katti/Desktop/Lab/stair/up_stair.jpg"

# 画像の読み込み
image = cv2.imread(image_path)

# 階段を検出
results = model.predict(source=image_path, imgsz=640)

# デバッグ用: 各結果オブジェクト内の情報を確認
for result in results:
    print(result)

# 検出結果の表示
for result in results:
    for box in result.boxes:
        # バウンディングボックスの座標を取得
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        # クラスIDを取得
        class_id = int(box.cls.cpu().numpy()[0])
        # 信頼度をスカラ値に変換
        confidence = box.conf.cpu().numpy()[0]
        # ラベルと信頼度を取得
        label = f"{result.names[class_id]} {confidence:.2f}"
        # ラベルの表示位置を調整
        y1 = y1 - 10 if y1 - 10 > 10 else y1 + 10
        # バウンディングボックスとラベルを描画
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 結果の表示
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""