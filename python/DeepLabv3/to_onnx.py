import torch
from torchvision.models.segmentation import deeplabv3_resnet50


# 学習済みモデルのロード（例: ResNet-50バックボーンのDeepLabv3）
model = deeplabv3_resnet50(weights=None, num_classes=4)

# 2. pthファイルのパスを指定
pth_path = "C:/Users/cpsla/PycharmProjects/segmentation/DeepLab/best_model_epoch50.pth"  # ここに.pthファイルのパスを指定
state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)  # map_locationを適宜設定


# state_dictから不要なキーを削除
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

# 3. モデルに重みをロード
#model.load_state_dict(state_dict)
# モデルにstate_dictをロード
model.load_state_dict(filtered_state_dict, strict=False)

# 推論モードに切り替え
model.eval()

# ダミー入力の作成（画像サイズ 3x512x512 のテンソルを例として使用）
dummy_input = torch.randn(1, 3, 512, 512)

# ONNX形式にエクスポート
torch.onnx.export(
    model,
    dummy_input,
    "deeplabv3.onnx",  # 出力先ファイル名
    input_names=["input"],  # ONNXモデルの入力名
    output_names=["output"],  # ONNXモデルの出力名
    opset_version=11  # 必要に応じて変更
)

print("ONNXファイルが 'deeplabv3.onnx' として保存されました。")