import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 保存ディレクトリの指定
output_dir = r'C:\Users\cpsla\PycharmProjects\intel_realsense\pictures'
os.makedirs(output_dir, exist_ok=True)

# RealSenseパイプラインの初期化
pipeline = rs.pipeline()
config = rs.config()

# 解像度とフレームレートの設定
width, height, fps = 640, 480, 30
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

# ストリーム開始
pipeline.start(config)

print("リアルタイムで深度画像を取得中... 's'キーで保存, 'q'キーで終了")

try:
    frame_count = 0
    while True:
        # フレームの取得
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # 深度画像の取得
        depth_image = np.asanyarray(depth_frame.get_data())

        # 深度画像をカラーマップに変換
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # 画像の表示
        cv2.imshow('RealSense Depth Stream', depth_colormap)

        # キー入力待機
        key = cv2.waitKey(1) & 0xFF

        # 's'キーで保存
        if key == ord('s'):
            filename = os.path.join(output_dir, f"depth_image_{frame_count:04d}.png")
            cv2.imwrite(filename, depth_colormap)
            print(f"深度画像を保存しました: {filename}")
            frame_count += 1

        # 'q'キーで終了
        if key == ord('q'):
            break

finally:
    # ストリーム停止
    pipeline.stop()
    cv2.destroyAllWindows()
    print("終了しました。")