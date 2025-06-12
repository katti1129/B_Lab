import json
import os
import math

def convert_labelme_to_houghnet(json_path, output_dir):
    """
    LabelMe JSONをHoughNetで使える形式に変換する
    """
    # 入力ファイルと出力ディレクトリの確認
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"指定されたJSONファイルが見つかりません: {json_path}")
    os.makedirs(output_dir, exist_ok=True)

    # JSONファイルを読み込む
    with open(json_path, 'r') as f:
        data = json.load(f)

    hough_lines = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'line':
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 始点と終点から直線のパラメータ (ρ, θ) を計算
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                raise ValueError("直線の始点と終点が同じ座標です")
            rho = abs(x1 * dy - y1 * dx) / math.sqrt(dx**2 + dy**2) if dx != 0 or dy != 0 else 0
            theta = math.atan2(dy, dx)

            hough_lines.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "rho": rho,
                "theta": theta
            })

    # 保存形式をカスタマイズ可能（例：JSON形式で保存）
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_path))[0] + "_hough.json")
    with open(output_file, 'w') as f:
        json.dump(hough_lines, f, indent=4)

# 使用例
convert_labelme_to_houghnet(
    json_path="/Users/katti/Desktop/dataset/val/before_labels/frame_0458.json",
    output_dir="/Users/katti/Desktop/dataset/val/labels"
)