import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread("/Users/katti/Desktop/frame_0470", cv2.IMREAD_GRAYSCALE)

# ノイズ除去（メディアンフィルタ）
image = cv2.medianBlur(image, 5)

# Sobelフィルタを適用
sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)               # 水平方向の勾配
sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)               # 垂直方向の勾配

# 勾配の絶対値を計算
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# 水平方向と垂直方向の勾配を組み合わせて合成勾配を計算
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# 画像とエッジ画像を表示
plt.rcParams["figure.figsize"] = [12,7.5]                           # ウィンドウサイズを設定
title = "cv2.Sobel: codevace.com"
plt.figure(title)                                                   # ウィンドウタイトルを設定
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)   # 余白を設定
plt.subplot(221)                                                    # 2行2列の1番目の領域にプロットを設定
plt.imshow(image, cmap='gray')                                      # 入力画像をグレースケールで表示
plt.title('Original Image')                                         # 画像タイトル設定
plt.axis("off")                                                     # 軸目盛、軸ラベルを消す
plt.subplot(222)                                                    # 2行2列の2番目の領域にプロットを設定
plt.imshow(sobel_x, cmap='gray')                                    # X方向のエッジ検出結果画像をグレースケールで表示
plt.title('Sobel X')                                                # 画像タイトル設定
plt.axis("off")                                                     # 軸目盛、軸ラベルを消す
plt.subplot(223)                                                    # 2行2列の3番目の領域にプロットを設定
plt.imshow(sobel_y, cmap='gray')                                    # Y方向のエッジ検出結果画像をグレースケールで表示
plt.title('Sobel Y')                                                # 画像タイトル設定
plt.axis("off")                                                     # 軸目盛、軸ラベルを消す
plt.subplot(224)                                                    # 2行2列の4番目の領域にプロットを設定
plt.imshow(sobel_combined, cmap='gray')                             # X方向のエッジ検出結果画像をグレースケールで表示
plt.title('Sobel Combined')                                         # 画像タイトル設定
plt.axis("off")                                                     # 軸目盛、軸ラベルを消す
plt.show()