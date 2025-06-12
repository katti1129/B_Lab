#モデルの内容を出力する。
import torch
model = torch.load("/Users/katti/Desktop/stepLR_epoch31.pth", map_location=torch.device('cpu'))
print(model)
