#一枚の画像のYOLOのバウンディングボックス内のマスク画像の割合とYOLOの信頼値の調和平均

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
yolo_model_path =  "C:/Users/cpsla/PycharmProjects/segmentation/DeepLab/model/best.pt" # 学習済みYOLOモデルのパス
yolo_model = YOLO(yolo_model_path)


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

"""
# 5. マスクの割合を計算する関数
def calculate_mask_ratios(mask, num_classes):
    total_pixels = mask.size
    class_ratios = {}

    for class_id in range(num_classes):
        class_pixels = np.sum(mask == class_id)
        class_ratios[class_id] = class_pixels / total_pixels * 100

    return class_ratios
"""


# 6. バウンディングボックス内のセグメンテーション割合を計算
def calculate_bbox_mask_ratio(mask, bbox, class_id):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_mask = mask[y1:y2, x1:x2]
    total_pixels = cropped_mask.size
    class_pixels = np.sum(cropped_mask == class_id)
    return class_pixels / total_pixels if total_pixels > 0 else 0


# 7. 実行例
if __name__ == "__main__":
    class_colors = [
        (255, 255, 0),  # 背景: 黄色
        (255, 0, 0),  # 下り階段: 赤
        (0, 255, 0),  # ステップ: 緑
        (0, 0, 255),  # 上り階段: 青
    ]

    image_path = "C:/Users/cpsla/PycharmProjects/segmentation/images/tmp/frame_0074.png"

    # YOLOで物体検出
    yolo_results = yolo_model(image_path)
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()



    # セグメンテーションマスクを予測
    predicted_mask = predict_mask(model, image_path, device)

    # バウンディングボックス内のセグメンテーション割合を計算し、結果を表示
    for bbox in yolo_boxes:
        print(f"YOLOのバウンディングボックスの大きさは",bbox)
        for class_id in range(1, num_classes):  # 背景以外のクラス
            ratio = calculate_bbox_mask_ratio(predicted_mask, bbox, class_id)
            print(ratio)
            if ratio >= 0.1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness = 10
                text = (f"confidence of YOLO&DeepLabv3:({ratio:.2f}%)")
                position = (10, 100)
                if class_id == 1:
                    print(f"下り階段を発見 ({ratio:.2f}%)")
                    color = (255, 0, 0)  # 赤色 (R, G, B)
                    # セグメンテーション結果を重ねた画像
                    segmentation_overlay = overlay_segmentation(image_path, predicted_mask, class_colors)
                    segmentation_overlay = np.array(segmentation_overlay)  # Pillowからnpに
                    cv2.putText(segmentation_overlay, text, position, font, font_scale, color, thickness)
                elif class_id == 2:
                    print(f"ステップを発見 ({ratio:.2f}%)")
                    color = (0, 255, 0)  # 赤色 (R, G, B)
                    # セグメンテーション結果を重ねた画像
                    segmentation_overlay = overlay_segmentation(image_path, predicted_mask, class_colors)
                    segmentation_overlay = np.array(segmentation_overlay)  # Pillowからnpに
                    cv2.putText(segmentation_overlay, text, position, font, font_scale, color, thickness)
                elif class_id == 3:
                    print(f"上り階段を発見 ({ratio:.2f}%)")
                    color = (0, 0, 255)  # 赤色 (R, G, B)
                    # セグメンテーション結果を重ねた画像
                    segmentation_overlay = overlay_segmentation(image_path, predicted_mask, class_colors)
                    segmentation_overlay = np.array(segmentation_overlay)#Pillowからnpに
                    cv2.putText(segmentation_overlay, text, position, font, font_scale, color, thickness)
                else:
                    segmentation_overlay = overlay_segmentation(image_path, predicted_mask, class_colors)


    # セグメンテーション結果を重ねた画像
    #segmentation_overlay = overlay_segmentation(image_path, predicted_mask, class_colors)

    # 画像をロード
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCVはBGR形式なのでRGBに変換

    # YOLOの検出結果からバウンディングボックスを描画
    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # バウンディングボックスの座標
        confidence = box.conf[0].item()  # 信頼度
        class_id = int(box.cls[0].item())  # クラスID

        # バウンディングボックスを描画
        if class_id == 0:#下り階段
            color = (255, 0, 0)  # 赤色 (R, G, B)
            thickness = 9
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # クラスIDと信頼度を描画
            label = f"Class: downstair, Conf: {confidence:.2f}"
            font_scale = 3.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1 + 30, y1 + 100), font, font_scale, color, thickness)
        if class_id == 1:#段差
            color = (0, 255, 0)  # 青色 (R, G, B)
            thickness = 9
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # クラスIDと信頼度を描画
            label = f"Class: step, Conf: {confidence:.2f}"
            font_scale = 3.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1 + 30, y1 + 100), font, font_scale, color, thickness)
        if class_id == 2:#上り階段
            color = (0, 0, 255)  # 青色 (R, G, B)
            thickness = 9
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # クラスIDと信頼度を描画
            label = f"Class: upstair, Conf: {confidence:.2f}"
            font_scale = 3.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1 + 30, y1 + 100), font, font_scale, color, thickness)

    # 結果を保存
    yolo_output_path = "C:/Users/cpsla/PycharmProjects/segmentation/runs/yolo_result.png"
    cv2.imwrite(yolo_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # 保存はBGR形式に戻す

    # 画像を表示
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("YOLO Detection Results")
    plt.imshow(Image.open(yolo_output_path))  # 保存したYOLO結果画像を読み込む

    plt.subplot(1, 2, 2)
    plt.title("Segmentation Overlay")
    plt.imshow(segmentation_overlay)

    plt.show()