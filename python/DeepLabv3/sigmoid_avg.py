import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import torchvision.models as models
from ultralytics import YOLO  # YOLOv8ライブラリ

# 1. DeepLabV3モデルの準備
num_classes = 4  # 背景(0) + 階段(1) + その他
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの初期化
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = models.segmentation.deeplabv3_resnet50(weights=weights)

# 出力クラス数に合わせてモデルの分類層を変更
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

# pthファイルのパスを指定して学習済みモデルをロード
pth_file_path = "C:/Users/cpsla/PycharmProjects/segmentation/DeepLab/model/CosineAnnealingLR_epoch43.pth"
model.load_state_dict(torch.load(pth_file_path, map_location=device))
model = model.to(device)
model.eval()

# 2. YOLOモデルの準備
yolo_model_path = "C:/Users/cpsla/PycharmProjects/segmentation/DeepLab/model/best.pt"  # 学習済みYOLOモデルのパス
yolo_model = YOLO(yolo_model_path)

# 全体の信頼度比率の合計とカウントをグローバル変数として初期化
global_confidence_ratio_sum = 0
global_count = 0

# 3. セグメンテーションマスクを予測する関数
def predict_mask(model, image_path, device, target_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize(target_size)


    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    predicted_mask = Image.fromarray(predicted_mask.astype(np.uint8))
    predicted_mask = predicted_mask.resize(original_size, Image.NEAREST)
    predicted_mask = np.array(predicted_mask)

    return predicted_mask

# 4. セグメンテーション結果を重ねる関数
def overlay_segmentation(image_path, mask, class_colors, alpha=0.6):
    original_image = Image.open(image_path).convert("RGB")
    original_image = np.array(original_image)

    color_mask = np.zeros_like(original_image, dtype=np.uint8)
    for class_id, color in enumerate(class_colors):
        color_mask[mask == class_id] = color

    blended_image = (original_image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return Image.fromarray(blended_image)

# 6. バウンディングボックス内のセグメンテーション割合を計算
def calculate_bbox_mask_ratio(model, image_path, device, bbox, yolo_class_id,target_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)

    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]  # モデル出力 (B, C, H, W)
        probs = torch.sigmoid(output)  # Sigmoidで確率に変換
        probs = probs.squeeze(0).cpu().numpy()  # (C, H, W) へ変換

        x1, y1, x2, y2 = map(int, bbox)
        #print(x1, y1, x2, y2)
        # bbox内の確率マップを抽出
        cropped_probs = probs[yolo_class_id, y1:y2, x1:x2]
        #print(cropped_probs.shape)

        if cropped_probs.size == 0:
            return 0.0  # 該当ピクセルがない場合は0を返す

        avg_prob = np.mean(cropped_probs)
        return avg_prob

# 5. 画像を処理して結果を保存する関数
def process_and_save_results(image_path, output_dir):
    global global_confidence_ratio_sum, global_count

    class_colors = [
        (255, 255, 0),  # 背景: 黄色0
        (255, 0, 0),  # 下り階段: 赤1
        (0, 255, 0),  # ステップ: 緑2
        (0, 0, 255),  # 上り階段: 青3
    ]
    # 画像をリサイズしてからYOLOで物体検出
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (512, 512))  # 画像をリサイズ
    # YOLOで物体検出
    yolo_results = yolo_model(image_resized)

    # セグメンテーションマスクを予測
    predicted_mask = predict_mask(model, image_path, device)

    # 検出物がない場合の処理
    if len(yolo_results[0].boxes) == 0:
        print("YOLO: No objects detected")
        confidence = 0
        global_confidence_ratio_sum += confidence
        global_count += 1  # 検出がなくてもカウントを増やす
        print("カウント数は:", global_count)

    segmentation_overlay = None
    # 通常の処理（YOLOが物体を検出した場合）
    for box in yolo_results[0].boxes:
        confidence = box.conf[0].item()
        bbox = box.xyxy[0].cpu().numpy()
        print("YOLOの信頼値:", confidence)

        for class_id in range(1, num_classes):  # 背景以外のクラス
            ratio = calculate_bbox_mask_ratio(model, image_path, device, bbox, class_id)
            print("クラスは : ",class_id,"マスク割合は : ",ratio)
            confidence_ratio = confidence * (0.5 + (0.5 * ratio)) if confidence != 0 else 0

            if class_id == 1 and box.cls == 0:
                print(f"下り階段を発見 ({confidence_ratio:.2f}%)")
                global_confidence_ratio_sum += confidence_ratio
                global_count += 1
                print("合計は:", global_confidence_ratio_sum)
                print("カウント数は:", global_count)
            elif class_id == 2 and box.cls == 1:
                print(f"ステップを発見 ({confidence_ratio:.2f}%)")
            elif class_id == 3 and box.cls == 2:
                print(f"上り階段を発見 ({confidence_ratio:.2f}%)")
                #global_confidence_ratio_sum += confidence_ratio
                #global_count += 1
                #print("合計は:", global_confidence_ratio_sum)
                # #print("カウント数は:", global_count)


            # セグメンテーション結果を重ねた画像を作成
            segmentation_overlay = overlay_segmentation(image_path, predicted_mask, class_colors)
            segmentation_overlay = np.array(segmentation_overlay)  # Pillowからnpに変換
            segmentation_overlay = Image.fromarray(segmentation_overlay)

    if segmentation_overlay is None:
        segmentation_overlay = overlay_segmentation(image_path, predicted_mask, class_colors)

    # セグメンテーション結果を重ねた画像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCVはBGR形式なのでRGBに変換

    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = box.conf[0].item()  # 信頼度
        class_id = int(box.cls[0].item())  # クラスID

        # バウンディングボックスを描画
        if class_id == 0:  # 下り階段
            color = (255, 0, 0)  # 赤色 (R, G, B)
            label = f"Class: downstair, Conf: {confidence:.2f}"
        elif class_id == 1:  # 段差
            color = (0, 255, 0)  # 緑色
            label = f"Class: step, Conf: {confidence:.2f}"
        elif class_id == 2:  # 上り階段
            color = (0, 0, 255)  # 青色
            label = f"Class: upstair, Conf: {confidence:.2f}"
        else:
            global_count += 1
            print("カウント数は:", global_count)
            continue

        thickness = 9
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        cv2.putText(image, label, (x1 + 30, y1 + 100), font, font_scale, color, thickness)

    # 結果を保存
    image_name = os.path.basename(image_path)
    yolo_output_path = os.path.join(output_dir, f"yolo_{image_name}")
    segmentation_output_path = os.path.join(output_dir, f"segmentation_{image_name}")

    cv2.imwrite(yolo_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    segmentation_overlay.save(segmentation_output_path)

    print(f"Results saved: {yolo_output_path}, {segmentation_output_path}")

# 7. 全体の平均値を出力
def print_global_average():
    global global_confidence_ratio_sum, global_count
    if global_count > 0:
        global_avg = global_confidence_ratio_sum / global_count
    else:
        global_avg = 0
    print(f"Global Average YOLO and DeepLabv3 confidence ratio: {global_avg:.2f}")

# 6. メイン処理
if __name__ == "__main__":
    input_dir = "C:/Users/cpsla/PycharmProjects/segmentation/images/inside_downstair"
    output_dir = "C:/Users/cpsla/PycharmProjects/segmentation/runs_23"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            process_and_save_results(image_path, output_dir)

print_global_average()