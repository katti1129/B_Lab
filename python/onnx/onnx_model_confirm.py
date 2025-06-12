import torch

# モデルの読み込み
model_path = "/Users/katti/Desktop/Lab/best.pt"
model = torch.load(model_path)

# モデルのアーキテクチャを確認
print(model)
