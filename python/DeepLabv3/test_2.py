#マスク画像とオリジナル画像おなじフォルダ内

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import torchvision.models as models
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(512, 512)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.images = [f for f in os.listdir(data_dir) if not f.endswith("_mask.png")]

        if not self.images:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.data_dir, img_file)
        mask_file = img_file.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.data_dir, mask_file)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Image or mask not found: {img_path} / {mask_path}")

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
        except Exception as e:
            raise IOError(f"Error loading image or mask: {e}")

        image = image.resize(self.target_size)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.long))

        return image, mask


def calculate_metrics(model, dataloader, device, num_classes):
    model.eval()
    precisions, recalls, f1s = [], [], []
    ious_per_class = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.numpy()

            # Per-image metrics
            precisions.append(precision_score(masks.flatten(), preds.flatten(),
                                              average="weighted", zero_division=0))
            recalls.append(recall_score(masks.flatten(), preds.flatten(),
                                        average="weighted", zero_division=0))
            f1s.append(f1_score(masks.flatten(), preds.flatten(),
                                average="weighted", zero_division=0))

            # Per-image IoU for each class
            for class_idx in range(num_classes):
                intersection = np.sum((masks == class_idx) & (preds == class_idx))
                union = np.sum((masks == class_idx) | (preds == class_idx))
                if union > 0:
                    ious_per_class[class_idx].append(intersection / union)

    # Calculate final metrics
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    mean_iou_per_class = [np.mean(ious) if len(ious) > 0 else float('nan')
                          for ious in ious_per_class]
    mean_iou = np.nanmean(mean_iou_per_class)

    return mean_precision, mean_recall, mean_f1, mean_iou_per_class, mean_iou


def main():
    # Configuration
    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "C:/Users/cpsla/PycharmProjects/segmentation/DeepLab/model/CosineAnnealingLR_epoch43.pth"
    test_data_dir = "C:/Users/cpsla/PycharmProjects/segmentation/dataset/final_segmentation/test"

    # Model setup
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    model = model.to(device)

    # Dataset setup
    test_dataset = SegmentationDataset(
        data_dir=test_data_dir,
        transform=transforms.ToTensor(),
        target_size=(512, 512)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Calculate metrics
    precision, recall, f1, iou_per_class, miou = calculate_metrics(
        model, test_loader, device, num_classes
    )

    # Print results
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Class-wise IoU:")
    for i, iou in enumerate(iou_per_class):
        print(f"Class {i}: {iou:.4f}")
    print(f"Mean IoU (mIoU): {miou:.4f}")


if __name__ == "__main__":
    main()