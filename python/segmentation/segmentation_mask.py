#COCOフォーマットのアノテーション（_annotations.coco.json）からセグメンテーションマスク画像を生成し、保存する

import json
import numpy as np
from PIL import Image, ImageDraw
import os

# COCOアノテーションファイルのパス
coco_annotation_path = "/Users/katti/Desktop/final_segmentation.v1i.coco-segmentation/train/_annotations.coco.json"
output_dir = "/Users/katti/Desktop/final_segmentation.v1i.coco-segmentation/masks"

# 出力フォルダを作成
os.makedirs(output_dir, exist_ok=True)

# JSONファイルを読み込む
with open(coco_annotation_path, 'r') as f:
    coco_data = json.load(f)

# 画像IDとそのサイズを取得
images_info = {img["id"]: (img["width"], img["height"], img["file_name"]) for img in coco_data["images"]}

# セグメンテーションマスクを生成
for annotation in coco_data["annotations"]:
    img_id = annotation["image_id"]
    segmentation = annotation["segmentation"]
    category_id = annotation["category_id"]

    # 画像の幅と高さ、ファイル名を取得
    width, height, file_name = images_info[img_id]

    # 空のマスク画像を作成
    mask = np.zeros((height, width), dtype=np.uint8)

    # ポリゴンを描画
    for polygon in segmentation:
        polygon = np.array(polygon).reshape((-1, 2))
        img = Image.fromarray(mask)
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in polygon], outline=category_id, fill=category_id)
        mask = np.array(img)

    # マスク画像のファイル名を元画像ファイル名に基づいて生成
    mask_file_name = file_name.replace(".jpg", "_mask.png")  # .jpg を _mask.png に置換

    # マスク画像をPNG形式で保存
    mask_image = Image.fromarray(mask)
    mask_image.save(os.path.join(output_dir, mask_file_name))

print("マスク画像の生成が完了しました。")
