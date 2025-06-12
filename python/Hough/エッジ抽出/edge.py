import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread("/Users/katti/Desktop/0115.jpg", cv2.IMREAD_GRAYSCALE)

# Cannyエッジ検出
edges = cv2.Canny(image, threshold1=101, threshold2=204)

# 結果の表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap="gray")
plt.show()
