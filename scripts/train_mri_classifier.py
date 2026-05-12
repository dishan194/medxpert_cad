"""
scripts/train_mri_classifier.py
Training script specifically for the Kaggle Brain MRI Tumor dataset.
Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Folder structure expected:
  data/raw/brain_mri/
    Training/
      glioma/
      meningioma/
      notumor/
      pituitary/
    Testing/
      glioma/
      meningioma/
      notumor/
      pituitary/

Usage:
    python scripts/train_mri_classifier.py --data-dir data/raw/brain_mri --epochs 20
"""

import argparse
import os
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Kaggle dataset class names (notumor maps to our "Normal" label)
KAGGLE_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_CLASSES  = ["Glioma", "Meningioma", "Normal", "Pituitary_Tumor"]


def get_transforms(image_size: int = 224, augment: bool = False):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.3),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet50 with custom head for MRI tumor classification."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze all except last 2 blocks + head
    for name, param in model.named_parameters():
        if "layer4" not in name and "layer3" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        if batch_idx % 20 == 0:
            logger.info(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                        f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")

    return {"loss": total_loss / len(loader), "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    logger.info("\n" + classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))

    return {
        "loss":     total_loss / len(loader),
        "accuracy": correct / total,
        "preds":    all_preds,
        "labels":   all_labels,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "Training"
    test_dir  = data_dir / "Testing"

    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        logger.error("Please place your Kaggle dataset at: data/raw/brain_mri/Training/")
        return

    # ImageFolder automatically maps subfolder names to class indices
    train_ds = datasets.ImageFolder(str(train_dir), transform=get_transforms(args.image_size, augment=True))
    test_ds  = datasets.ImageFolder(str(test_dir),  transform=get_transforms(args.image_size, augment=False))

    logger.info(f"Classes found: {train_ds.classes}")
    logger.info(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    # Class weights for imbalanced data
    class_counts = [0] * len(train_ds.classes)
    for _, label in train_ds.samples:
        class_counts[label] += 1
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model     = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics   = evaluate(model, test_loader, criterion, device, train_ds.classes)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']*100:.2f}% | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% | Time: {elapsed:.1f}s"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            save_path = save_dir / "mri_classifier.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved best model → {save_path} (Acc: {best_acc*100:.2f}%)")

            # Also save class mapping for inference
            import json
            mapping = {
                "class_to_idx": train_ds.class_to_idx,
                "idx_to_class": {v: k for k, v in train_ds.class_to_idx.items()},
                "accuracy": best_acc,
            }
            with open(save_dir / "mri_class_mapping.json", "w") as f:
                json.dump(mapping, f, indent=2)

    logger.info(f"\nTraining complete. Best Test Accuracy: {best_acc*100:.2f}%")
    logger.info(f"Model saved to: {save_dir / 'mri_classifier.pth'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Brain MRI Tumor Classifier (Kaggle dataset)")
    p.add_argument("--data-dir",    type=str, default="data/raw/brain_mri")
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--image-size",  type=int, default=224)
    p.add_argument("--save-dir",    type=str, default="data/models")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)