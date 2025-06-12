import numpy as np
from PIL import Image

# 画像の読み込み
mask_image = Image.open("/Users/katti/Desktop/steps50.v1i.coco/mask/kaidan4_jpg_mask.png")
mask_array = np.array(mask_image)

# ピクセル値のユニークな値を取得
unique_values = np.unique(mask_array)
print("Unique pixel values in the mask:", unique_values)
