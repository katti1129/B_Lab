import pyrealsense2 as rs
import time

# パイプラインを初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

# ストリーミング開始
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # ポイントクラウドを作成
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)

        # PLYファイル保存用の保存オプションを設定
        save_path = f"./ply/5/frame_{int(time.time())}.ply"
        ply = rs.save_to_ply(save_path)
        ply.set_option(rs.save_to_ply.option_ply_binary, False)  # テキスト形式で保存q
        ply.set_option(rs.save_to_ply.option_ply_normals, True)  # 法線情報を含む

        # ポイントクラウドを保存
        ply.process(depth_frame)
        print(f"Saved PLY file: {save_path}")

        # インターバルを設定（例: 1秒）
        time.sleep(1)

finally:
    # ストリーミング終了
    pipeline.stop()
