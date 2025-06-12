import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込む（グレースケール）
image_path ="/Users/katti/Desktop/frame_0470.png"  # 画像のパスを変更してください
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError("指定された画像が見つかりません。パスを確認してください。")

# FFTの計算
dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(dft)  # 中心を画像の中央に移動
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)  # 振幅スペクトル

# 元の画像とFFTスペクトルをプロット
plt.figure(figsize=(12, 6))

# 元の画像
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray, cmap="gray")
plt.axis("off")

# FFTスペクトル
plt.subplot(1, 2, 2)
plt.title("FFT Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
