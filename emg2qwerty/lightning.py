# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TransformerEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=self.window_length,
                    padding=(1800, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
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
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TransformerModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        transformer_embed_dim: int | None = None,
        transformer_num_heads: int = 4,
        transformer_layers: int = 2,
        transformer_drop_prob: float = 0.1,
        transformer_max_len: int = 4096,
        transformer_dim_feedforward: int | None = None,
        decoder_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if decoder_chunk_size is not None:
            assert decoder_chunk_size > 0
        self.decoder_chunk_size = decoder_chunk_size

        num_features = self.NUM_BANDS * mlp_features[-1]
        transformer_embed_dim = (
            transformer_embed_dim
            if transformer_embed_dim is not None
            else num_features
        )

        _ = block_channels
        _ = kernel_width

        # Inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.frontend = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            # (T, N, transformer_embed_dim)
            nn.Linear(num_features, transformer_embed_dim),
        )

        self.encoder = TransformerEncoder(
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            layers=transformer_layers,
            drop_prob=transformer_drop_prob,
            max_len=transformer_max_len,
            dim_feedforward=transformer_dim_feedforward,
        )

        self.classifier = nn.Sequential(
            # (T, N, num_classes)
            nn.Linear(transformer_embed_dim, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.frontend(inputs)
        x = self.encoder(x, input_lengths=input_lengths)
        return self.classifier(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs, input_lengths=input_lengths)

        # This architecture preserves temporal length.
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        emission_np = emissions.detach().cpu().numpy()
        emission_len_np = emission_lengths.detach().cpu().numpy()
        if self.decoder_chunk_size is not None and phase in {"val", "test"}:
            predictions = self.decoder.decode_batch_chunked(
                emissions=emission_np,
                emission_lengths=emission_len_np,
                chunk_size=self.decoder_chunk_size,
            )
        else:
            predictions = self.decoder.decode_batch(
                emissions=emission_np,
                emission_lengths=emission_len_np,
            )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(
            f"{phase}/loss",
            loss,
            batch_size=N,
            sync_dist=True,
            on_epoch=True,
            prog_bar=phase == "val",
        )
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        computed = metrics.compute()
        for metric_name, metric_value in computed.items():
            self.log(
                metric_name,
                metric_value,
                sync_dist=True,
                prog_bar=phase == "val",
            )
        if phase == "val":
            val_loss = self.trainer.callback_metrics.get("val/loss")
            val_loss_str = (
                f"val/loss={float(val_loss):.4f}"
                if val_loss is not None
                else "val/loss=nan"
            )
            self.print(
                " | ".join(
                    [
                        val_loss_str,
                        f"val/CER={float(computed['val/CER']):.4f}",
                        f"val/IER={float(computed['val/IER']):.4f}",
                        f"val/DER={float(computed['val/DER']):.4f}",
                        f"val/SER={float(computed['val/SER']):.4f}",
                    ]
                )
            )
        metrics.reset()

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

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
