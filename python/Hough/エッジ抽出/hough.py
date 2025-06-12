import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込みframe_0217.png
image = cv2.imread("/Users/katti/Desktop/0115.jpg")  # 入力画像のパス
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


median_intensity = np.median(gray)  # グレースケール画像の中央値
lower_threshold = int(max(0, 0.66 * median_intensity))
upper_threshold = int(min(255, 1.33 * median_intensity))
print(lower_threshold,upper_threshold)
edges = cv2.Canny(gray, lower_threshold, upper_threshold)

# エッジ検出 (Canny)
#edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Hough変換で線を検出
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=300)
# Hough変換で線を検出
#lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150, minLineLength=50, maxLineGap=10)

# Hough変換が失敗した場合
if lines is None:
    print("No lines detected.")
    exit()

# 平行線を抽出
parallel_lines = []  # 初期化
angle_threshold = 5 * np.pi / 180  # 平行条件: 角度差が5度以内

for i, line1 in enumerate(lines):
    rho1, theta1 = line1[0]
    group = [line1]  # 新しいグループを作成
    for j, line2 in enumerate(lines):
        if i != j:
            rho2, theta2 = line2[0]
            if abs(theta1 - theta2) < angle_threshold:  # 平行条件をチェック
                group.append(line2)
    if len(group) > 1:  # 平行線が複数ある場合のみグループ化
        parallel_lines.append(group)

# 平行線が見つからなかった場合の処理
if not parallel_lines:
    print("No parallel lines detected.")
    exit()

# 平行線を描画
output_image = image.copy()
for group in parallel_lines:
    for line in group:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # 直線の長さを計算
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # 長さが一定以上の場合のみ描画
        min_length = 1500  # 最小線分長さの閾値
        if length > min_length:
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青色で描画

# 結果を表示
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
