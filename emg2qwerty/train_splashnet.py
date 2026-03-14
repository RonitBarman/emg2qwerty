#!/usr/bin/env python3
"""
Training script for SplashNet on EMG2QWERTY data using CTC loss.

Replicates the SplashNet architecture training procedure with:
  - Fourier-transform-based preprocessing (LogSpectrogram)
  - CTC loss for sequence-to-sequence learning
  - Train/val/test splits from config/user/single_user.yaml
  - Checkpointing with resume support

Usage:
    python train_splashnet.py [--args]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.model import SplashNet
from emg2qwerty.transforms import (
    Compose,
    ForEach,
    LogSpectrogram,
    RandomBandRotation,
    SpecAugment,
    TemporalAlignmentJitter,
    ToTensor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Inline CTC greedy decoder (avoids kenlm dependency from decoder.py)
# ─────────────────────────────────────────────────────────────────────────────
class CTCGreedyDecoder:
    """Minimal greedy CTC decoder that collapses repeated labels and
    removes blanks.  Produces a list of LabelData, one per batch item."""

    def __init__(self):
        self._charset = charset()

    def decode_batch(
        self,
        emissions: np.ndarray,
        emission_lengths: np.ndarray,
    ) -> list[LabelData]:
        assert emissions.ndim == 3  # (T, N, num_classes)
        N = emissions.shape[1]
        decodings = []
        for i in range(N):
            labels: list[int] = []
            prev = self._charset.null_class
            for t in range(int(emission_lengths[i])):
                label = int(emissions[t, i].argmax())
                if label != self._charset.null_class and label != prev:
                    labels.append(label)
                prev = label
            decodings.append(
                LabelData.from_labels(labels, _charset=self._charset)
            )
        return decodings


log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test splits from config/user/single_user.yaml
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SESSIONS = [
    "2021-06-03-1622765527-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-02-1622681518-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-04-1622863166-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-22-1627003020-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-21-1626916256-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-22-1627004019-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-05-1622885888-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-03-1622764398-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-21-1626917264-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-05-1622889105-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-03-1622766673-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-04-1622861066-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-22-1627001995-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-05-1622884635-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-21-1626915176-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
]

VAL_SESSIONS = [
    "2021-06-04-1622862148-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
]

TEST_SESSIONS = [
    "2021-06-02-1622682789-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
]


# ─────────────────────────────────────────────────────────────────────────────
# Transforms (same preprocessing pipeline as the original codebase)
# ─────────────────────────────────────────────────────────────────────────────
def build_train_transform() -> Compose:
    """Training transform: ToTensor → BandRotation → TemporalJitter
    → LogSpectrogram → SpecAugment."""
    return Compose(
        [
            ToTensor(fields=("emg_left", "emg_right")),
            ForEach(RandomBandRotation(offsets=(-1, 0, 1))),
            TemporalAlignmentJitter(max_offset=120),
            LogSpectrogram(n_fft=64, hop_length=16),
            SpecAugment(
                n_time_masks=3,
                time_mask_param=25,
                n_freq_masks=2,
                freq_mask_param=4,
            ),
        ]
    )


def build_eval_transform() -> Compose:
    """Validation / test transform: ToTensor → LogSpectrogram."""
    return Compose(
        [
            ToTensor(fields=("emg_left", "emg_right")),
            LogSpectrogram(n_fft=64, hop_length=16),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Module
# ─────────────────────────────────────────────────────────────────────────────
class SplashNetDataModule(pl.LightningDataModule):
    """Lightning DataModule that wraps WindowedEMGDataset for SplashNet
    training with the same windowing, padding, and collation."""

    def __init__(
        self,
        data_root: Path,
        train_sessions: list[str],
        val_sessions: list[str],
        test_sessions: list[str],
        window_length: int = 8000,
        padding: tuple[int, int] = (1800, 200),
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.train_paths = [self.data_root / f"{s}.hdf5" for s in train_sessions]
        self.val_paths = [self.data_root / f"{s}.hdf5" for s in val_sessions]
        self.test_paths = [self.data_root / f"{s}.hdf5" for s in test_sessions]

        self.window_length = window_length
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path=p,
                    transform=build_train_transform(),
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for p in self.train_paths
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path=p,
                    transform=build_eval_transform(),
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for p in self.val_paths
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path=p,
                    transform=build_eval_transform(),
                    # Full session at test time (no windowing)
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for p in self.test_paths
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────
class SplashNetCTCModule(pl.LightningModule):
    """PyTorch Lightning module wrapping SplashNet with CTC loss,
    greedy decoding, and CER metrics."""

    NUM_BANDS = 2
    ELECTRODE_CHANNELS = 16

    def __init__(
        self,
        input_freqs: int = 33,
        hidden_dim: int = 384,
        num_blocks: int = 4,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        max_epochs: int = 150,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_classes = charset().num_classes
        self.model = SplashNet(
            num_classes=num_classes,
            input_freqs=input_freqs,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )

        # CTC loss (blank = null_class, which is the last index)
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

        # Greedy CTC decoder
        self.decoder = CTCGreedyDecoder()

        # Per-phase metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            left:  (B, T, C=16, freq)
            right: (B, T, C=16, freq)
        Returns:
            log_probs: (T, B, num_classes)   – time-first for CTC
        """
        logits = self.model(left, right)  # (B, T, num_classes)
        log_probs = logits.log_softmax(dim=-1)  # (B, T, num_classes)
        return log_probs.permute(1, 0, 2)  # (T, B, num_classes)

    # ── shared step ──────────────────────────────────────────────────────
    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]  # (T, N, bands=2, C=16, freq)
        targets = batch["targets"]  # (T_label, N)
        input_lengths = batch["input_lengths"]  # (N,)
        target_lengths = batch["target_lengths"]  # (N,)
        N = len(input_lengths)

        # Split left / right bands and convert to (N, T, C, freq)
        left = inputs[:, :, 0, :, :].permute(1, 0, 2, 3)   # (N, T, 16, freq)
        right = inputs[:, :, 1, :, :].permute(1, 0, 2, 3)   # (N, T, 16, freq)

        # Forward pass → log-probs (T, N, num_classes)
        emissions = self.forward(left, right)

        # SplashNet uses causal padding so temporal length is preserved:
        # emission T == input T.  Compute emission lengths accordingly.
        emission_lengths = input_lengths.clone()

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (N, T_label)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Greedy decoding + metrics (on CPU)
        emissions_np = emissions.detach().cpu().numpy()
        emission_lengths_np = emission_lengths.detach().cpu().numpy()
        predictions = self.decoder.decode_batch(emissions_np, emission_lengths_np)

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        # Training: show per-step loss in progress bar.
        # Validation/test: log epoch-level only (avoids PL _step/_epoch suffixes).
        is_train = (phase == "train")
        self.log(
            f"{phase}/loss",
            loss,
            batch_size=N,
            sync_dist=True,
            prog_bar=is_train,
            on_step=is_train,
            on_epoch=True,
        )
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        computed = metrics.compute()

        # Guard against ZeroDivisionError when no targets were seen
        # (e.g. sanity check batches with all-empty labels).
        safe = {}
        for k, v in computed.items():
            if isinstance(v, float) and (v != v or v == float("inf")):
                v = 0.0  # replace nan/inf with 0
            # PL 1.8.x log_dict works most reliably with tensors
            safe[k] = torch.tensor(v) if not isinstance(v, torch.Tensor) else v

        if safe:
            self.log_dict(safe, sync_dist=True)

        # Print human-readable summary so metrics are clearly visible
        # (the progress bar often truncates long metric names).
        loss_key = f"{phase}/loss"
        cer_key = f"{phase}/CER"
        loss_val = self.trainer.callback_metrics.get(loss_key)
        cer_val = safe.get(cer_key)
        loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
        cer_str = f"{cer_val:.2f}%" if cer_val is not None else "N/A"
        print(
            f"  ► [{phase.upper()}] Epoch {self.current_epoch}  "
            f"loss={loss_str}  CER={cer_str}"
        )

        metrics.reset()

    # ── Lightning hooks ──────────────────────────────────────────────────
    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    # ── Optimizer & LR scheduler ─────────────────────────────────────────
    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Linear warmup + cosine annealing (same schedule as the baseline)
        warmup_epochs = self.hparams.warmup_epochs
        max_epochs = self.hparams.max_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup from ~0 → 1
                return max(1e-5, epoch / max(1, warmup_epochs))
            # Cosine annealing from 1 → eta_min/lr
            eta_min_ratio = 1e-6 / self.hparams.lr
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (
                1.0 + np.cos(np.pi * progress)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the most recently modified .ckpt file in the directory."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SplashNet with CTC loss")

    # Data
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/jeffrey/emg2qwerty/emg2qwerty/single_user/89335547"),
        help="Directory containing the .hdf5 session files",
    )

    # Model
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--input-freqs", type=int, default=33,
                    help="Number of frequency bins (n_fft//2 + 1)")

    # Training
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1501)

    # Windowing
    p.add_argument("--window-length", type=int, default=8000,
                    help="Window length in raw EMG samples (2 kHz)")
    p.add_argument("--left-padding", type=int, default=1800,
                    help="Left contextual padding in raw EMG samples")
    p.add_argument("--right-padding", type=int, default=200,
                    help="Right contextual padding in raw EMG samples")

    # Checkpoints / logging
    p.add_argument("--output-dir", type=Path,
                    default=Path("splashnet_runs"),
                    help="Root directory for checkpoints and logs")
    p.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in output-dir")

    # Hardware
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Seed
    pl.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    # ── Data ─────────────────────────────────────────────────────────────
    datamodule = SplashNetDataModule(
        data_root=args.data_root,
        train_sessions=TRAIN_SESSIONS,
        val_sessions=VAL_SESSIONS,
        test_sessions=TEST_SESSIONS,
        window_length=args.window_length,
        padding=(args.left_padding, args.right_padding),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────
    module = SplashNetCTCModule(
        input_freqs=args.input_freqs,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
    )
    log.info(
        f"SplashNet — hidden_dim={args.hidden_dim}, blocks={args.num_blocks}, "
        f"num_classes={charset().num_classes}"
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="splashnet-{epoch:03d}-{val/CER:.4f}",
            monitor="val/CER",
            mode="min",
            save_last=True,
            save_top_k=3,
            verbose=True,
        ),
    ]

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    # ── Resume from checkpoint if requested ──────────────────────────────
    resume_ckpt = None
    if args.resume:
        resume_ckpt = get_last_checkpoint(checkpoint_dir)
        if resume_ckpt is not None:
            log.info(f"Resuming training from {resume_ckpt}")
        else:
            log.info("No checkpoint found to resume from — starting fresh")

    # ── Train ────────────────────────────────────────────────────────────
    trainer.fit(module, datamodule, ckpt_path=resume_ckpt)

    # ── Load best checkpoint for evaluation ──────────────────────────────
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        log.info(f"Loading best checkpoint: {best_ckpt}")
        module = SplashNetCTCModule.load_from_checkpoint(best_ckpt)

    # ── Validate & Test ──────────────────────────────────────────────────
    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    log.info(f"Validation metrics: {val_metrics}")
    log.info(f"Test metrics:       {test_metrics}")
    log.info(f"Best checkpoint:    {best_ckpt}")


if __name__ == "__main__":
    main()
