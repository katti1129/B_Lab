import cv2
import numpy as np
from ultralytics import YOLO  # YOLOv8を使用する例

# 1. 画像を読み込む
image = cv2.imread("/Users/katti/Desktop/0115.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. YOLOで階段候補を検出
model = YOLO("/Users/katti/Desktop/Lab/LabPython/best.pt")  # カスタム階段モデル
results = model(image)


# YOLO推論結果を解析
for result in results:  # 複数の結果に対応
    if hasattr(result, "boxes") and result.boxes is not None:
        for box in result.boxes.xyxy:  # バウンディングボックスを取得
            x1, y1, x2, y2 = map(int, box)  # 座標を整数に変換
            roi = image[y1:y2, x1:x2]  # 検出領域を切り出し

            # グレースケール変換
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 明るさとコントラストを調整（正規化）
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # ノイズ除去
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Sobelフィルターをy軸方向に適用
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
            sobel_y = cv2.convertScaleAbs(sobel_y)  # 結果を絶対値に変換してスケーリング

            median_intensity = np.median(gray)  # グレースケール画像の中央値
            lower_threshold = int(max(0, 0.66 * median_intensity))
            upper_threshold = int(min(255, 1.33 * median_intensity))
            print(lower_threshold,upper_threshold)
            # ハフ変換を適用
            #edges = cv2.Canny(gray, lower_threshold, upper_threshold)  # Cannyでエッジを強調
            edges = cv2.Canny(sobel_y, lower_threshold, upper_threshold)  # Cannyでエッジを強調
            lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

            # 検出された線を描画
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # バウンディングボックスを描画
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 結果の表示
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
