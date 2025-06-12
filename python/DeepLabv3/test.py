#mask画像とimages画像それぞれフォルダが分かれている

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# === 設定 ===
MODEL_PATH = "C:/Users/cpsla/PycharmProjects/segmentation/DeepLab/best_model_epoch50.pth"  # 学習済みモデルのパス
TEST_IMAGE_DIR = "C:/Users/cpsla/PycharmProjects/segmentation/dataset/final_segmentation.v1i.png-mask-semantic/test/images"  # テスト画像ディレクトリ
OUTPUT_DIR = "C:/Users/cpsla/PycharmProjects/segmentation/dataset/final_segmentation.v1i.png-mask-semantic/test/masks"  # マスク出力ディレクトリ
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 設定 ===
NUM_CLASSES = 4  # 学習時のクラス数

# === モデルの読み込みと調整 ===
print("Loading model...")
# DeepLabv3モデルを定義
model = models.segmentation.deeplabv3_resnet50(weights=None)

# 出力クラス数を設定
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))

# 学習済みの重みをロード
checkpoint = torch.load(MODEL_PATH)

# `aux_classifier` を削除
filtered_state_dict = {k: v for k, v in checkpoint.items() if "aux_classifier" not in k}

# 重みをモデルにロード
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()  # 評価モードに切り替え
print("Model loaded successfully.")

"""
# === モデルの読み込み ===
print("Loading model...")
model = models.segmentation.deeplabv3_resnet50(weights=None)  # モデルを定義
model.load_state_dict(torch.load(MODEL_PATH))  # 学習済み重みを読み込む
model.eval()  # 評価モードに切り替え
print("Model loaded successfully.")
"""

# === 前処理 ===
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # モデルに合わせたサイズにリサイズ
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
])

# === 予測とマスク保存関数 ===
def predict_and_save_mask(image_path, output_dir, model):
    # 入力画像の読み込み
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # バッチ次元を追加

    # モデルによる予測
    with torch.no_grad():
        output = model(input_tensor)["out"]
        predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # ピクセルごとのクラス予測

    # マスクを保存
    output_mask_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_mask.png"))
    mask_image = Image.fromarray(predicted_mask.astype(np.uint8))
    mask_image.save(output_mask_path)

    return predicted_mask

# === テスト画像に対する予測と結果表示 ===
print("Processing test images...")
for image_file in os.listdir(TEST_IMAGE_DIR):
    if image_file.endswith((".jpg", ".png")):  # 画像ファイルのみ対象
        image_path = os.path.join(TEST_IMAGE_DIR, image_file)
        print(f"Processing: {image_file}")

        # マスクの生成と保存
        predicted_mask = predict_and_save_mask(image_path, OUTPUT_DIR, model)

        # 可視化（任意）
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(image_path))
        plt.title("Input Image")
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap="jet")
        plt.title("Predicted Mask")
        plt.show()

print("All test images processed and masks saved.")
