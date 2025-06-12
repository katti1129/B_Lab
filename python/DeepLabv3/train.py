import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from tqdm import tqdm  # tqdmライブラリをインポート


num_classes = 4  # クラス数(背景を含めた)
patience = 10  # 検証損失が改善しないエポック数の許容範囲
num_epochs = 100  # 学習エポック数


# 1. データセットクラスの定義
class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(512, 512)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.images = [f for f in os.listdir(data_dir) if not f.endswith("_mask.png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.data_dir, img_file)

        # 対応するマスクファイル名を生成
        mask_file = img_file.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.data_dir, mask_file)

        # 画像とマスクの読み込み
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # リサイズ
        image = image.resize(self.target_size)
        mask = mask.resize(self.target_size, Image.NEAREST)

        # 変換
        if self.transform:
            image = self.transform(image)
        mask = np.array(mask, dtype=np.int64)  # マスクは整数型に変換
        mask = torch.from_numpy(mask)

        return image, mask

# 2. 評価指標の計算関数

def calculate_accuracy(outputs, masks):
    preds = outputs.argmax(dim=1)
    correct = (preds == masks).sum().item()
    total = masks.numel()
    return correct / total

def calculate_iou(outputs, masks, num_classes):
    preds = outputs.argmax(dim=1)
    iou_per_class = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (masks == cls)).sum().item()
        union = ((preds == cls) | (masks == cls)).sum().item()
        if union == 0:
            iou_per_class.append(float('nan'))  # クラスが画像内に存在しない場合
        else:
            iou_per_class.append(intersection / union)
    return np.nanmean(iou_per_class)

# 3. データセットのパス設定とデータローダー
train_dataset = SegmentationDataset(
    data_dir="C:/Users/cpsla/PycharmProjects/segmentation/dataset/final_segmentation.v2i.png-mask-semantic/train",
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ]),
    target_size=(512, 512)
)

val_dataset = SegmentationDataset(
    data_dir="C:/Users/cpsla/PycharmProjects/segmentation/dataset/final_segmentation.v2i.png-mask-semantic/valid",
    transform=transforms.ToTensor(),
    target_size=(512, 512)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 4. DeepLabv3+ モデルの準備
#num_classes = 4  # 背景(0) + 階段(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# デバイスに応じたメッセージを出力
if device.type == "cuda":
    print("CUDAが使用されています")
else:
    print("CPUが使用されています")

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = models.segmentation.deeplabv3_resnet50(weights=weights)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
model = model.to(device)

# 5. 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss(ignore_index=255)  # 未使用ラベル(255)を無視
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習率スケジューラー（10エポックごとに学習率を*0.8に）
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

# 早期終了のパラメータ
#patience = 10  # 検証損失が改善しないエポック数の許容範囲
best_val_loss = float('inf')
patience_counter = 0

# 6. 学習ループ
#num_epochs = 50  # 学習エポック数
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_accuracy = 0.0
    train_iou = 0.0

    # tqdmを用いて進捗バーを作成
    with tqdm(total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]", ncols=80) as pbar:
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            # **デバイス確認用のコードを追加**
            if batch_idx == 0:  # 1バッチ目のみ確認するよう制限
                print(f"Images are on: {images.device}")
                print(f"Masks are on: {masks.device}")
            optimizer.zero_grad()

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_accuracy += calculate_accuracy(outputs, masks)
            train_iou += calculate_iou(outputs, masks, num_classes)

            # バッチごとの進捗を更新
            pbar.update(1)
            pbar.set_postfix({"Batch Loss": loss.item()})

    # エポックごとの平均損失、精度、IoUを計算
    avg_train_loss = running_loss / len(train_loader)
    avg_train_accuracy = train_accuracy / len(train_loader)
    avg_train_iou = train_iou / len(train_loader)
    print(f"\nEpoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Train IoU: {avg_train_iou:.4f}")

    # 検証ループ
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, masks)
            val_iou += calculate_iou(outputs, masks, num_classes)

    # 検証データの平均損失、精度、IoUを計算
    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}, Validation IoU: {avg_val_iou:.4f}")

    # 現在の学習率を取得して表示
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning Rate: {current_lr:.8f}")
    # 学習率の更新
    scheduler.step()
    # 現在の学習率を取得して表示
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning Rate: {current_lr:.8f}")

    # 早期終了のチェック
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # モデルを保存
        torch.save(model.state_dict(), f"best_model_2_epoch{epoch + 1}.pth")
        print("\nモデルを保存しました！")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("\n早期終了条件を満たしました。学習を終了します。")
            break

print("学習が完了しました！")
