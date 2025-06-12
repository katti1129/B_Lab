import pyrealsense2 as rs
import numpy as np
import cv2

# RealSenseパイプラインの設定
pipeline = rs.pipeline()
config = rs.config()
# デプスストリームを設定
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    # パイプラインの開始
    pipeline.start(config)

    while True:
        # フレームの取得
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # 深度フレームをnumpy配列に変換
        depth_image = np.asanyarray(depth_frame.get_data())

        # 中央の四角形の座標を定義
        center_x, center_y = depth_image.shape[1] // 2, depth_image.shape[0] // 2
        square_size = 50
        x_start, y_start = center_x - square_size // 2, center_y - square_size // 2

        # 四角形内のピクセルのZ軸の平均値を計算
        square_depth_values = depth_image[y_start:y_start + square_size, x_start:x_start + square_size]
        average_depth = np.mean(square_depth_values) * depth_frame.get_units()

        # 深度画像をカラーマップに変換
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 四角形の描画
        cv2.rectangle(depth_colormap, (x_start, y_start), (x_start + square_size, y_start + square_size), (255, 0, 0),
                      2)

        # 中央の四角形の平均Z軸の高さを表示
        cv2.putText(depth_colormap, f"{average_depth:.2f}m", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

        # 画像の表示
        cv2.imshow("Depth Image with Rectangle", depth_colormap)

        # キー入力待ち
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
