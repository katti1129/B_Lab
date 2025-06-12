import pyrealsense2 as rs
import open3d as o3d
import numpy as np

# RealSenseパイプラインの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Open3Dビジュアライザ設定
vis = o3d.visualization.Visualizer()
vis.create_window()

try:
    while True:
        # フレーム取得
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # 深度データをnumpy配列に変換し、スケールを調整
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # メートル単位に変換

        print("a")

        # Open3Dでポイントクラウド生成
        depth_o3d = o3d.geometry.Image(depth_image)

        print("b")

        # 内部パラメータ取得
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

        print("c")
        # 深度イメージからポイントクラウド生成
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)
        )

        print("d")
        # 平面検出
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"平面方程式: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        print("e")

        # 平面を抽出して表示
        plane_cloud = pcd.select_by_index(inliers)
        vis.clear_geometries()
        vis.add_geometry(plane_cloud)
        vis.poll_events()
        vis.update_renderer()
        print("f")

except KeyboardInterrupt:
    print("終了します。")

finally:
    pipeline.stop()
    vis.destroy_window()
