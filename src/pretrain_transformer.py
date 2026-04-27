from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader, Dataset

from src.qm9_data import QM9_TARGETS, QM9Normalizer, get_qm9_splits
from src.transformer_model import (
    CHEMBERTA_MODEL,
    QM9PretrainHead,
    SMILESTransformer,
    tokenize,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class QM9Dataset(Dataset):
    """
    PyTorch Dataset wrapping a QM9 DataFrame split.

    Tokenization is performed once at construction time (the tokenizer is
    CPU-only and cheap to cache) to avoid repeated tokenization during
    training.  Normalised targets are stored as a float32 tensor.
    """

    def __init__(
        self,
        df,
        targets: list[str],
        normalizer: QM9Normalizer,
        max_length: int = 128,
    ) -> None:
        """
        Tokenize SMILES and normalise targets up front.

        Params:
            df: pd.DataFrame : split DataFrame with 'Drug' and target columns
            targets: list[str] : QM9 target column names to include
            normalizer: QM9Normalizer : fitted normaliser
            max_length: int : tokenizer max sequence length
        Returns:
            None
        """
        enc = tokenize(df["Drug"].tolist(), max_length=max_length)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(
            normalizer.transform(df, targets), dtype=torch.float32
        )

    def __len__(self) -> int:
        """
        Return dataset size.

        Params:
            None
        Returns:
            int : number of molecules
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return tokenized inputs and normalised targets for one molecule.

        Params:
            idx: int : sample index
        Returns:
            dict[str, torch.Tensor] : 'input_ids', 'attention_mask', 'labels'
        """
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class PretrainLitModule(L.LightningModule):
    """Lightning wrapper for multi-task QM9 pretraining of SMILESTransformer."""

    def __init__(
        self,
        model: SMILESTransformer,
        targets: list[str],
        normalizer: QM9Normalizer,
        lr: float = 2e-5,
        weight_decay: float = 1e-2,
        warmup_steps: int = 500,
        t_max: int = 10,
        qm9_split: str = "stratified_scaffold",
    ) -> None:
        """
        Wrap SMILESTransformer for QM9 multi-task regression.

        Params:
            model: SMILESTransformer : model with QM9PretrainHead attached
            targets: list[str] : QM9 target names being predicted
            normalizer: QM9Normalizer : fitted normaliser, embedded in checkpoint
            lr: float : peak AdamW learning rate
            weight_decay: float : AdamW weight decay
            warmup_steps: int : linear warmup steps before cosine decay
            t_max: int : CosineAnnealingLR period in epochs
            qm9_split: str : split strategy used for pretraining, embedded in checkpoint
        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.targets = targets
        self.normalizer = normalizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.t_max = t_max
        self.qm9_split = qm9_split
        self.loss_fn = nn.MSELoss()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Embed targets and normaliser state in the checkpoint.

        Params:
            checkpoint: dict : Lightning checkpoint dict (mutated in place)
        Returns:
            None
        """
        checkpoint["pretrain_targets"] = self.targets
        checkpoint["normalizer_state"] = self.normalizer.state_dict()
        checkpoint["qm9_split"] = self.qm9_split

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        preds = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(preds, batch["labels"])
        # Per-target MAE for interpretability in wandb
        per_target_mae = (preds - batch["labels"]).abs().mean(dim=0)
        self.log(
            f"{split}_loss",
            loss,
            prog_bar=(split == "val"),
            batch_size=len(batch["labels"]),
        )
        for name, mae in zip(self.targets, per_target_mae):
            self.log(f"{split}_mae_{name}", mae, batch_size=len(batch["labels"]))
        return loss

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """
        Compute multi-task MSE loss for a training batch.

        Params:
            batch: Any : dict with input_ids, attention_mask, labels
            _batch_idx: int : batch index (unused)
        Returns:
            torch.Tensor : scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Any, _batch_idx: int) -> None:
        """
        Log validation multi-task MSE and per-target MAE.

        Params:
            batch: Any : dict with input_ids, attention_mask, labels
            _batch_idx: int : batch index (unused)
        Returns:
            None
        """
        self._step(batch, "val")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Return AdamW with linear warmup and cosine annealing.

        Params:
            None
        Returns:
            dict : Lightning optimizer/scheduler config
        """
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Linear warmup scheduler chained with cosine decay via SequentialLR
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=self.warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def pretrain(
    config: dict | None = None,
    checkpoint_dir: str | Path = "checkpoints/pretrain",
) -> SMILESTransformer:
    """
    Pretrain SMILESTransformer on QM9 multi-task regression.

    Params:
        config: dict | None : hyperparameters; missing keys fall back to defaults
        checkpoint_dir: str | Path : directory for Lightning checkpoints
    Returns:
        SMILESTransformer : best pretrained model with backbone weights updated
    """
    cfg = {
        "targets": QM9_TARGETS,
        "seed": 42,
        "batch_size": 128,
        "max_epochs": 10,
        "patience": 3,
        "lr": 2e-5,
        "weight_decay": 1e-2,
        "warmup_steps": 500,
        "max_length": 128,
        "d_hidden_head": 768,  # unused for pretrain head but kept for consistency
        "qm9_split": "stratified_scaffold",
        "wandb_project": "lipophilicity_pred",
        "wandb_run_name": None,
    }
    if config:
        cfg.update(config)

    targets: list[str] = cfg["targets"]

    L.seed_everything(cfg["seed"], workers=True)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    splits = get_qm9_splits(seed=cfg["seed"], method=cfg["qm9_split"])
    normalizer = QM9Normalizer().fit(splits["train"], targets)

    train_ds = QM9Dataset(splits["train"], targets, normalizer, cfg["max_length"])
    val_ds = QM9Dataset(splits["valid"], targets, normalizer, cfg["max_length"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model ---
    head = QM9PretrainHead(d_model=768, n_targets=len(targets))
    model = SMILESTransformer(head=head, model_name=CHEMBERTA_MODEL)

    # --- Loggers ---
    wandb_logger = WandbLogger(
        project=cfg["wandb_project"],
        name=cfg.get("wandb_run_name"),
        config={**cfg, "phase": "pretrain"},
    )
    csv_logger = CSVLogger(save_dir=str(ckpt_dir), name="logs")
    run_name = wandb_logger.experiment.name

    # --- Callbacks ---
    early_stop = EarlyStopping(monitor="val_loss", patience=cfg["patience"], mode="min")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name}-pretrain-{{epoch:03d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    lit = PretrainLitModule(
        model,
        targets=targets,
        normalizer=normalizer,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        t_max=cfg["max_epochs"],
        qm9_split=cfg["qm9_split"],
    )

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="auto",
        callbacks=[early_stop, ckpt_cb],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=[wandb_logger, csv_logger],
    )
    trainer.fit(lit, train_loader, val_loader)

    import shutil

    history_src = Path(csv_logger.log_dir) / "metrics.csv"
    history_dst = ckpt_dir / "pretrain_metrics_history.csv"
    if history_src.exists():
        shutil.copy(history_src, history_dst)

    return ckpt_cb.best_model_path


def load_pretrained_backbone(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[SMILESTransformer, list[str], QM9Normalizer]:
    """
    Restore a pretrained SMILESTransformer backbone from a Lightning checkpoint.

    The returned model has no head attached — call swap_head() before use.
    The pretrain targets and normaliser are also returned for inspection.

    Params:
        checkpoint_path: str | Path : path to the .ckpt file from pretrain()
        device: torch.device | None : target device; defaults to CUDA if available
    Returns:
        tuple : (SMILESTransformer with head removed, target list, QM9Normalizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    targets = ckpt["pretrain_targets"]
    normalizer = QM9Normalizer.from_state_dict(ckpt["normalizer_state"])

    # Rebuild model architecture to match the saved weights
    head = QM9PretrainHead(d_model=768, n_targets=len(targets))
    model = SMILESTransformer(head=head, model_name=CHEMBERTA_MODEL)

    # Strip the "model." prefix added by Lightning
    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state)
    model = model.to(device).eval()

    return model, targets, normalizer
