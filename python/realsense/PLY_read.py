import open3d as o3d
import numpy as np
from pyransac3d import Plane

# 点群データ読み込み
try:
    pcd_load = o3d.io.read_point_cloud("./frame_1731591006.ply")
    print("Point cloud loaded successfully.")
except Exception as e:
    print("Error loading point cloud:", e)

# numpy配列に変換
points = np.asarray(pcd_load.points)
print("Points array shape:", points.shape)

# 平面モデルを定義
plano1 = Plane()

# RANSACによる平面推定。しきい値は0.01
try:
    best_eq, best_inliers = plano1.fit(points, 0.01)
    print("Plane equation:", best_eq)
    print("Number of inliers:", len(best_inliers))
except Exception as e:
    print("Error in RANSAC plane fitting:", e)


print(best_inliers)
# 元のデータにおけるインライアの点の色を変更
try:
    plane = pcd_load.select_by_index(best_inliers).paint_uniform_color([1, 0, 0])
    print("Inliers colored.")
except Exception as e:
    print("Error coloring inliers:", e)

# 平面のバウンディングボックスを取得
try:
    obb = plane.get_oriented_bounding_box()
    obb.color = [0, 0, 1]
    print("Oriented bounding box created.")
except Exception as e:
    print("Error creating bounding box:", e)

# 平面以外の点を抽出
try:
    not_plane = pcd_load.select_by_index(best_inliers, invert=True)
    print("Non-plane points extracted.")
except Exception as e:
    print("Error extracting non-plane points:", e)

# Open3Dで可視化
try:
    o3d.visualization.draw_geometries([not_plane, plane, obb])
    print("Visualization complete.")
except Exception as e:
    print("Error during visualization:", e)
