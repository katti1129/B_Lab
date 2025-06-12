import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2

# RealSenseパイプラインの初期化
pipeline = rs.pipeline()
config = rs.config()
width = 1024
height = 768
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

# ストリーム開始
profile = pipeline.start(config)

# デバイスとセンサーの取得
device = profile.get_device()
depth_sensor = device.first_depth_sensor()

# 測定距離を伸ばす設定
if depth_sensor.supports(rs.option.visual_preset):
    depth_sensor.set_option(rs.option.visual_preset, 5)  # 最大範囲プリセット
if depth_sensor.supports(rs.option.laser_power):
    depth_sensor.set_option(rs.option.laser_power, 100)  # 最大値（200）

# 深度データを保存するリスト
depth_values = []

try:
    print("リアルタイムで深度を取得中... (Ctrl+Cで停止)")
    while True:
        # フレーム取得
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # デプス画像を取得
        depth_image = np.asanyarray(depth_frame.get_data())

        # 中心点の深度取得 (x=width//2, y=height//2)
        center_x, center_y = width // 2, height // 2
        center_depth = depth_frame.get_distance(center_x, center_y)
        print(f"中心点の深度: {center_depth:.2f} m")

        # デプス画像をカラーマップに変換
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # 中心点に円を描画
        cv2.circle(depth_colormap, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

        # カラーマップ画像を表示
        cv2.imshow('RealSense Depth Stream', depth_colormap)

        # 深度値をリストに追加
        depth_values.append(center_depth)

        # キー入力待機 (1ms) / 'q'で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("停止しました。深度値をプロットします。")
finally:
    # ストリーム停止
    pipeline.stop()
    cv2.destroyAllWindows()

# グラフ化
plt.figure(figsize=(10, 5))
plt.plot(depth_values, marker='o')
plt.title("Change in depth value of center point", fontsize=16)
plt.xlabel("Frame", fontsize=16)
plt.ylabel("Depth (m)", fontsize=16)
plt.grid()
plt.show()