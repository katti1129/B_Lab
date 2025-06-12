import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

# デバイス情報を取得
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# サポートされるストリーム設定を表示
print("サポートされるストリーム解像度とフレームレート:")
for sensor in device.sensors:
    if sensor.get_info(rs.camera_info.name) == "Stereo Module":
        for profile in sensor.profiles:
            video_profile = profile.as_video_stream_profile()
            print(f"  - {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()} FPS")