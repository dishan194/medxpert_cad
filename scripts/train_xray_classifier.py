"""
scripts/train_xray_classifier.py
Training script for the multi-label chest X-ray classifier.
Supports ResNet50 and ViT. Uses NIH ChestX-ray14 dataset.

Usage:
    python scripts/train_xray_classifier.py \
        --data-dir data/raw/nih_chestxray \
        --model-type resnet50 \
        --epochs 30 \
        --batch-size 32
"""

import argparse
import os
import csv
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


class NIHChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 dataset.
    Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC
    """

    def __init__(self, data_dir: str, split_file: str, transform=None, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.samples: List[Tuple[str, np.ndarray]] = []

        self._load_labels(split_file)

    def _load_labels(self, split_file: str):
        """Load image paths and multi-hot labels from CSV."""
        with open(split_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.data_dir / "images" / row["Image Index"]
                if not img_path.exists():
                    continue

                # Parse findings string → multi-hot vector
                findings = row.get("Finding Labels", "No Finding").split("|")
                label_vec = np.zeros(len(PATHOLOGY_LABELS), dtype=np.float32)
                for finding in findings:
                    if finding in PATHOLOGY_LABELS:
                        label_vec[PATHOLOGY_LABELS.index(finding)] = 1.0

                self.samples.append((str(img_path), label_vec))

        logger.info(f"Loaded {len(self.samples)} samples from {split_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(label)

    def get_class_weights(self) -> torch.Tensor:
        """Compute positive class weights for imbalanced multi-label."""
        all_labels = np.array([s[1] for s in self.samples])
        pos_counts = all_labels.sum(axis=0)
        neg_counts = len(self.samples) - pos_counts
        weights = neg_counts / (pos_counts + 1e-7)
        return torch.from_numpy(weights).float()


def get_transforms(image_size: int = 224, augment: bool = False):
    """Build image transforms for train/val."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )

    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """BCE loss with per-class positive weights for class imbalance."""

    def __init__(self, pos_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weights", pos_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights.to(logits.device)
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

        if batch_idx % 50 == 0:
            logger.info(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Mean AUC across classes (skip if single-class batch)
    try:
        aucs = [
            roc_auc_score(all_targets[:, i], all_preds[:, i])
            for i in range(all_targets.shape[1])
            if all_targets[:, i].sum() > 0
        ]
        mean_auc = float(np.mean(aucs)) if aucs else 0.0
    except Exception:
        mean_auc = 0.0

    return {
        "loss":     total_loss / len(loader),
        "mean_auc": mean_auc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Per-class AUC
    aucs = {}
    for i, label in enumerate(PATHOLOGY_LABELS):
        if all_targets[:, i].sum() > 0:
            try:
                aucs[label] = roc_auc_score(all_targets[:, i], all_preds[:, i])
            except Exception:
                aucs[label] = 0.0

    mean_auc = float(np.mean(list(aucs.values()))) if aucs else 0.0

    # Threshold at 0.5 for F1
    preds_binary = (all_preds >= 0.5).astype(int)
    f1_macro = f1_score(all_targets, preds_binary, average="macro", zero_division=0)

    return {
        "loss":       total_loss / len(loader),
        "mean_auc":   mean_auc,
        "f1_macro":   f1_macro,
        "per_class_auc": aucs,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Datasets
    train_ds = NIHChestXrayDataset(
        args.data_dir,
        args.train_csv or str(Path(args.data_dir) / "train_list.txt"),
        transform=get_transforms(args.image_size, augment=True),
        augment=True,
    )
    val_ds = NIHChestXrayDataset(
        args.data_dir,
        args.val_csv or str(Path(args.data_dir) / "val_list.txt"),
        transform=get_transforms(args.image_size, augment=False),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model
    from backend.models.xray_classifier import ResNet50Classifier, ViTClassifier
    if args.model_type == "vit":
        model = ViTClassifier(num_classes=len(PATHOLOGY_LABELS))
    else:
        model = ResNet50Classifier(num_classes=len(PATHOLOGY_LABELS))

    model = model.to(device)

    # Loss with class weights
    pos_weights = train_ds.get_class_weights().to(device)
    criterion   = WeightedBCELoss(pos_weights)

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_auc   = 0.0
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['mean_auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['mean_auc']:.4f} "
            f"F1: {val_metrics['f1_macro']:.4f} | Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_metrics["mean_auc"] > best_auc:
            best_auc = val_metrics["mean_auc"]
            save_path = save_dir / f"xray_{args.model_type}_best.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved best model → {save_path} (AUC: {best_auc:.4f})")

        # Save per-class AUC breakdown
        if epoch % 5 == 0:
            logger.info("  Per-class AUC:")
            for label, auc in sorted(val_metrics["per_class_auc"].items(), key=lambda x: -x[1]):
                logger.info(f"    {label:25s}: {auc:.4f}")

    logger.info(f"Training complete. Best Val AUC: {best_auc:.4f}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train chest X-ray classifier")
    parser.add_argument("--data-dir",   type=str, required=True,       help="NIH ChestX-ray14 directory")
    parser.add_argument("--train-csv",  type=str, default=None)
    parser.add_argument("--val-csv",    type=str, default=None)
    parser.add_argument("--model-type", type=str, default="resnet50",   choices=["resnet50", "vit"])
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-dir",   type=str, default="data/models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)