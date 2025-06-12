import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread("/Users/katti/Desktop/frame_0470.png")  # 入力画像のパス
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 平滑化処理（ノイズ除去）
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Sobelフィルタで横方向のエッジを強調
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=0, ksize=3)  # 横方向エッジ
sobel_x_abs = cv2.convertScaleAbs(sobel_x)  # 絶対値を取り、8ビットに変換

# エッジ検出
edges = cv2.Canny(sobel_x_abs, 100, 150)

# Hough変換で線を検出
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

# 水平線のみを抽出
horizontal_lines = []  # 水平線を格納するリスト
angle_threshold = 5 * np.pi / 180  # 角度の許容範囲: ±10度

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        # θ が 0 (水平) または π (180度) に近い場合のみ選択
        if abs(theta - 0) < angle_threshold or abs(theta - np.pi) < angle_threshold:
            horizontal_lines.append((rho, theta))

# 水平線が見つからなかった場合の処理
if not horizontal_lines:
    print("No horizontal lines detected.")
    exit()

# 水平線を描画
output_image = image.copy()
for rho, theta in horizontal_lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青色で描画

# 結果を表示
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title("Sobel + Canny Edges")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Horizontal Lines")
plt.axis('off')

plt.tight_layout()
plt.show()
