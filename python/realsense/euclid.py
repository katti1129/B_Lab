import pyrealsense2 as rs
import numpy as np
import cv2
import math

# RealSenseパイプラインの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# クリックしたポイントを保存するためのリスト
points = []


# マウスクリックイベントの処理
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:
            calculate_distance(points)
            points = []  # 次のクリックでリセット


# 2点間の距離を計算する関数
def calculate_distance(points):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    # 各点の深度データを取得
    depth1 = depth_frame.get_distance(points[0][0], points[0][1])
    depth2 = depth_frame.get_distance(points[1][0], points[1][1])

    # カメラの内部パラメータを取得
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # ピクセル座標を3D空間の座標に変換
    point1_3d = rs.rs2_deproject_pixel_to_point(intrinsics, points[0], depth1)
    point2_3d = rs.rs2_deproject_pixel_to_point(intrinsics, points[1], depth2)

    # 2点間のユークリッド距離を計算
    distance = math.sqrt(sum([(c1 - c2) ** 2 for c1, c2 in zip(point1_3d, point2_3d)]))
    print(f"2点間の距離: {distance:.2f} meters")


try:
    # ウィンドウとマウスイベントの設定
    cv2.namedWindow("Depth Image")
    cv2.setMouseCallback("Depth Image", mouse_callback)

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        # 深度画像の表示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Depth Image", depth_colormap)

        # qキーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
