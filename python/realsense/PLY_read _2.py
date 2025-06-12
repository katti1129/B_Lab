import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

ply_point_cloud = o3d.data.PLYPointCloud() # デモデータの読込み
pcd = o3d.io.read_point_cloud("./ply/5/frame_1732182164.ply") # PLYファイルの読込み
o3d.visualization.draw_geometries([pcd]) # 表示
