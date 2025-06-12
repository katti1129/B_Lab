import open3d as o3d
import numpy as np

# PLYファイルを読み込む
pcd = o3d.io.read_point_cloud("./ply/5/frame_1732182164.ply")

# 点群情報を表示
print(pcd)

# 座標と色が含まれているか確認
print("点の数:", np.asarray(pcd.points).shape[0])
if pcd.has_colors():
    print("色情報が含まれています。")
else:
    print("色情報は含まれていません。")