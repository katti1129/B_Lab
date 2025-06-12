import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math

# RealSenseパイプラインの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# YOLOモデルのロード
model = YOLO("yolov8x.pt")  # 適切なモデルを指定


def calculate_euclidean_distance(depth_frame, bbox, intrinsics):
    """バウンディングボックス内の平均深度を計算してユークリッド距離を取得"""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # 中心の深度値を取得
    depth = depth_frame.get_distance(center_x, center_y)

    # 3D座標に変換
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth)
    distance = math.sqrt(point_3d[0] ** 2 + point_3d[1] ** 2 + point_3d[2] ** 2)
    return distance


try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # カラーフレームをYOLOに渡す
        color_image = np.asanyarray(color_frame.get_data())
        results = model(color_image)

        # RealSenseの内部パラメータを取得
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 検出された物体に対して処理
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
                label = result.names[int(box.cls.numpy()[0])]

                if label == "person":  # 特定のクラスに限定（例：人）
                    distance = calculate_euclidean_distance(depth_frame, (x1, y1, x2, y2), intrinsics)
                    print(f"{label}: 距離 {distance:.2f}m")

        # 検出結果を描画
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO with Depth", annotated_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
