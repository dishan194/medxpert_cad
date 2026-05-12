"""
scripts/train_report_generator.py
Training script for the encoder-decoder clinical report generator.
Uses IU X-Ray dataset (Indiana University Chest X-Ray Collection).

Download: https://openi.nlm.nih.gov/

Usage:
    python scripts/train_report_generator.py \
        --data-dir data/raw/iu_xray \
        --epochs 50 \
        --batch-size 16
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

from backend.models.report_generator import (
    VisualEncoder, ReportDecoder, MedicalVocabulary, ReportGenerator
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class IUXrayDataset(Dataset):
    """
    Indiana University Chest X-Ray dataset.
    Expected structure:
      data/raw/iu_xray/
        images/    (JPEG files)
        reports.json  (list of {image_id, findings, impression})
    """

    def __init__(
        self,
        data_dir: str,
        vocab: MedicalVocabulary,
        split: str = "train",
        max_seq_len: int = 128,
        image_size: int = 224,
    ):
        self.data_dir    = Path(data_dir)
        self.vocab       = vocab
        self.max_seq_len = max_seq_len
        self.split       = split
        self.samples: List[Dict] = []

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self._load_data()

    def _load_data(self):
        report_file = self.data_dir / "reports.json"
        if not report_file.exists():
            logger.warning(f"reports.json not found at {report_file}. Creating empty dataset.")
            return

        with open(report_file) as f:
            all_reports = json.load(f)

        # Simple 80/10/10 split
        n = len(all_reports)
        if self.split == "train":
            reports = all_reports[:int(0.8 * n)]
        elif self.split == "val":
            reports = all_reports[int(0.8 * n): int(0.9 * n)]
        else:
            reports = all_reports[int(0.9 * n):]

        for item in reports:
            img_path = self.data_dir / "images" / item["image_id"]
            if not img_path.exists():
                continue
            # Combine findings + impression as the target report
            report_text = item.get("findings", "") + " " + item.get("impression", "")
            report_text = report_text.strip()

            # Build vocabulary
            for word in self.vocab._tokenize(report_text):
                self.vocab.add_word(word)

            self.samples.append({
                "image_path":  str(img_path),
                "report_text": report_text,
            })

        logger.info(f"[{self.split}] Loaded {len(self.samples)} samples. Vocab size: {self.vocab.size}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        token_ids  = self.vocab.encode(sample["report_text"], self.max_seq_len)
        caption    = torch.tensor(token_ids, dtype=torch.long)
        cap_length = len(token_ids)

        return image, caption, cap_length


def collate_fn(batch):
    """Pad captions to same length within batch."""
    images, captions, lengths = zip(*batch)
    images = torch.stack(images, 0)
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, (cap, length) in enumerate(zip(captions, lengths)):
        padded[i, :length] = cap[:length]
    return images, padded, torch.tensor(lengths, dtype=torch.long)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_epoch(
    encoder: VisualEncoder,
    decoder: ReportDecoder,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for batch_idx, (images, captions, lengths) in enumerate(loader):
        images   = images.to(device)
        captions = captions.to(device)
        targets  = captions[:, 1:]   # Shift right for teacher forcing

        encoder_out = encoder(images)
        predictions, _ = decoder(encoder_out, captions[:, :-1], lengths - 1)

        # Mask padding
        B, T, V = predictions.shape
        preds_flat   = predictions.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        loss = criterion(preds_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 20 == 0:
            logger.info(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_bleu(
    encoder: VisualEncoder,
    decoder: ReportDecoder,
    dataset: IUXrayDataset,
    device: torch.device,
    num_samples: int = 100,
) -> Dict[str, float]:
    """Evaluate BLEU-4 score on a subset."""
    encoder.eval()
    decoder.eval()

    references, hypotheses = [], []
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        image, _, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)
        encoder_out = encoder(image)

        token_ids = decoder.generate(
            encoder_out,
            sos_idx=dataset.vocab.sos_idx,
            eos_idx=dataset.vocab.eos_idx,
            max_len=128,
            beam_size=5,
        )

        hypothesis = dataset.vocab.decode(token_ids)
        reference  = dataset.samples[idx]["report_text"]

        references.append(reference)
        hypotheses.append(hypothesis)

    # Compute BLEU
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smoother = SmoothingFunction().method4
    ref_tokens = [[r.split()] for r in references]
    hyp_tokens = [h.split()   for h in hypotheses]

    bleu4 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25,)*4, smoothing_function=smoother)
    return {"bleu4": round(bleu4, 4)}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training report generator on: {device}")

    vocab = MedicalVocabulary()

    # Build datasets (vocab is populated during train_ds init)
    train_ds = IUXrayDataset(args.data_dir, vocab, split="train",
                             max_seq_len=args.max_seq_len, image_size=args.image_size)
    val_ds   = IUXrayDataset(args.data_dir, vocab, split="val",
                             max_seq_len=args.max_seq_len, image_size=args.image_size)

    if len(train_ds) == 0:
        logger.error("Training dataset is empty. Check data directory and reports.json.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)

    encoder  = VisualEncoder(encoded_image_size=14, encoder_dim=2048).to(device)
    decoder  = ReportDecoder(vocab_size=vocab.size, embed_dim=256,
                             decoder_dim=512, encoder_dim=2048).to(device)

    # Ignore padding index in loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    encoder_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr
    )
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.decoder_lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_bleu = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            encoder, decoder, train_loader,
            decoder_optimizer, criterion, device, epoch
        )

        bleu_scores = evaluate_bleu(encoder, decoder, val_ds, device)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"BLEU-4: {bleu_scores['bleu4']:.4f} | "
            f"SOTA: {bleu_scores['bleu4'] >= 0.415} | "
            f"Time: {elapsed:.1f}s"
        )

        if bleu_scores["bleu4"] > best_bleu:
            best_bleu = bleu_scores["bleu4"]
            checkpoint = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab":   vocab,
                "bleu4":   best_bleu,
                "epoch":   epoch,
            }
            save_path = save_dir / "report_generator_best.pth"
            torch.save(checkpoint, save_path)
            logger.info(f"  ✓ Saved best model → {save_path} (BLEU-4: {best_bleu:.4f})")

    logger.info(f"Training complete. Best BLEU-4: {best_bleu:.4f} (SOTA ≥ 0.415)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    type=str, required=True)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--image-size",  type=int, default=224)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--encoder-lr",  type=float, default=1e-4)
    p.add_argument("--decoder-lr",  type=float, default=4e-4)
    p.add_argument("--save-dir",    type=str, default="data/models")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)