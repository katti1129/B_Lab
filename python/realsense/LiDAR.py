import pyrealsense2 as rs
import numpy as np
import ransac_2 as o3d
import cv2


# Realsense設定の初期化
pipeline = rs.pipeline()
config = rs.config()

# デプスストリームを設定
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
profile = pipeline.start(config)

# 深度スケール（メートル単位）
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# 段差閾値（10cm）
threshold_distance = 0.1

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # 深度データを取得し、メートル単位に変換
        depth_image = np.asanyarray(depth_frame.get_data())

        # Open3Dのポイントクラウド形式に変換
        depth_o3d = o3d.geometry.Image(depth_image)

        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        )

        #print("a")
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, pinhole_camera_intrinsic, depth_scale=1.0 / depth_scale, stride=1
        )
        #print("a")

        # RANSACを用いた平面検出
        plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01,
                                                         ransac_n=3,
                                                         num_iterations=1000)

        # 平面に属さないポイント（外れ値）を取得
        outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

        # 段差の深度差が閾値を超えるかチェック
        if np.any(np.abs(np.asarray(outlier_cloud.points)[:, 2] - plane_model[3]) > threshold_distance):
            print("警告: 段差を検出しました！")

        # 可視化用にデータを表示（任意）
        o3d.visualization.draw_geometries([point_cloud])

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
