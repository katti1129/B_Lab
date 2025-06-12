import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# 1. DeepLabV3モデルの準備
num_classes = 4  # 背景(0) + 上り階段(1) + 下り階段(2) + その他(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの初期化
weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
model = models.segmentation.deeplabv3_resnet50(weights=weights)

# 出力クラス数に合わせてモデルの分類層を変更
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

# pthファイルのパスを指定して学習済みモデルをロード
pth_file_path = "C:/Users/cpsla/PycharmProjects/segmentation/DeepLab/model/CosineAnnealingLR_epoch43.pth"
model.load_state_dict(torch.load(pth_file_path, map_location=device))
model = model.to(device)
model.eval()

# 2. 画像の前処理
image_path = "C:/Users/cpsla/PycharmProjects/segmentation/images/tmp/frame_0003.png"
image = Image.open(image_path).convert("RGB")

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensor_image = transform(image).unsqueeze(0).to(device)  # バッチ次元を追加

# 3. 推論の実行
with torch.no_grad():
    output = model(tensor_image)['out']  # モデルの出力取得

# 4. Sigmoid関数を適用して確率を計算
sigmoid = nn.Sigmoid()
probabilities = sigmoid(output)

# 5. 画像の中央ピクセルの確率を取得
h, w = probabilities.shape[2], probabilities.shape[3]  # 高さと幅
center_h, center_w = h // 2, w // 2  # 画像の中心座標

center_probs = probabilities[0, :, center_h, center_w]  # 各クラスの確率

# 6. 結果を表示
labels = ["背景", "下り階段", "ステップ", "上り階段"]
for label, prob in zip(labels, center_probs.cpu().numpy()):
    print(f"{label}: {prob:.4f}")
