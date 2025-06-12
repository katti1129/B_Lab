import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# RealSenseカメラの初期設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# YOLOモデルの読み込み
model = YOLO('yolov8x.pt')  # 自分の学習済みモデルのパスを指定

try:
    while True:
        # RealSenseからカラー画像を取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Numpy配列に変換
        color_image = np.asanyarray(color_frame.get_data())

        # YOLOで物体検出
        results = model(color_image)

        # 検出結果の描画
        for result in results:
            for box in result.boxes:
                # バウンディングボックスの取得
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]  # 信頼度

                # バウンディングボックスを描画
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ラベル（クラス名と信頼度）を描画
                label = f"{box.cls} {confidence:.2f}"
                cv2.putText(color_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # カメラ映像を表示
        cv2.imshow('RealSense Object Detection', color_image)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # ストリーミング停止
    pipeline.stop()
    cv2.destroyAllWindows()
