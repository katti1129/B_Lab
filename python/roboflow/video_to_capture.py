import cv2
import os

# 動画ファイルのパス
video_path = "/Users/katti/Desktop/IMG_1463.MOV"

# 出力ディレクトリ
output_dir = "/Users/katti/Desktop/私立図書館_upstair"
os.makedirs(output_dir, exist_ok=True)

# 動画の読み込み
cap = cv2.VideoCapture(video_path)

# フレーム数のカウント
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームを画像として保存
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
print(f'動画から{frame_count}フレームを抽出しました。')