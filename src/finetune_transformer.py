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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from src.data import get_tdc_split
from src.pretrain_transformer import load_pretrained_backbone
from src.transformer_model import (
    CHEMBERTA_MODEL,
    LogDFinetuneHead,
    SMILESTransformer,
    tokenize,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LogDDataset(Dataset):
    """
    PyTorch Dataset wrapping a logD DataFrame split.

    Tokenization is performed once at construction to avoid re-tokenizing on
    every batch.
    """

    def __init__(self, df, max_length: int = 128) -> None:
        """
        Tokenize SMILES and store logD targets.

        Params:
            df: pd.DataFrame : split DataFrame with 'Drug' and 'Y' columns
            max_length: int : tokenizer max sequence length
        Returns:
            None
        """
        enc = tokenize(df["Drug"].tolist(), max_length=max_length)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(df["Y"].values, dtype=torch.float32)

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
        Return tokenized inputs and logD target for one molecule.

        Params:
            idx: int : sample index
        Returns:
            dict[str, torch.Tensor] : 'input_ids', 'attention_mask', 'label'
        """
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class FinetuneLitModule(L.LightningModule):
    """Lightning wrapper for logD fine-tuning of SMILESTransformer."""

    def __init__(
        self,
        model: SMILESTransformer,
        lr: float = 1e-5,
        weight_decay: float = 1e-2,
        warmup_steps: int = 100,
        t_max: int = 50,
        head_kwargs: dict | None = None,
        pretrained_checkpoint: str | None = None,
        pretrain_targets: list[str] | None = None,
    ) -> None:
        """
        Wrap SMILESTransformer for logD regression.

        Params:
            model: SMILESTransformer : model with LogDFinetuneHead attached
            lr: float : peak AdamW learning rate
            weight_decay: float : AdamW weight decay
            warmup_steps: int : linear warmup steps before cosine decay
            t_max: int : CosineAnnealingLR period in epochs
            head_kwargs: dict | None : LogDFinetuneHead constructor kwargs,
                embedded in checkpoint for self-contained loading
            pretrained_checkpoint: str | None : path to phase-1 checkpoint,
                embedded for traceability
            pretrain_targets: list[str] | None : QM9 targets used in phase 1,
                embedded for ablation tracking
        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.t_max = t_max
        self.loss_fn = nn.MSELoss()
        self._head_kwargs = head_kwargs or {}
        self._pretrained_checkpoint = pretrained_checkpoint
        self._pretrain_targets = pretrain_targets or []

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Embed head kwargs and provenance in the checkpoint.

        Params:
            checkpoint: dict : Lightning checkpoint dict (mutated in place)
        Returns:
            None
        """
        checkpoint["head_kwargs"] = self._head_kwargs
        checkpoint["pretrained_checkpoint"] = self._pretrained_checkpoint
        checkpoint["pretrain_targets"] = self._pretrain_targets

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        """
        Compute MSE loss and log MAE for a single logD batch.

        Params:
            batch: dict : dict with 'input_ids', 'attention_mask', 'label'
            split: str : one of 'train', 'val'; used as metric-name prefix
        Returns:
            torch.Tensor : scalar MSE loss
        """
        preds = self.model(batch["input_ids"], batch["attention_mask"]).squeeze(-1)
        targets = batch["label"]
        loss = self.loss_fn(preds, targets)
        mae = (preds - targets).abs().mean()
        self.log(
            f"{split}_loss", loss, prog_bar=(split == "val"), batch_size=len(targets)
        )
        self.log(
            f"{split}_mae", mae, prog_bar=(split == "val"), batch_size=len(targets)
        )
        return loss

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """
        Compute MSE loss for a training batch.

        Params:
            batch: Any : dict with input_ids, attention_mask, label
            _batch_idx: int : batch index (unused)
        Returns:
            torch.Tensor : scalar MSE loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Any, _batch_idx: int) -> None:
        """
        Log validation MAE and loss.

        Params:
            batch: Any : dict with input_ids, attention_mask, label
            _batch_idx: int : batch index (unused)
        Returns:
            None
        """
        self._step(batch, "val")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Return AdamW with linear warmup and cosine annealing.

        A lower learning rate than pretraining is used to avoid destroying the
        pretrained representations during fine-tuning.

        Params:
            None
        Returns:
            dict : Lightning optimizer/scheduler config
        """
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate_transformer(
    model: SMILESTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Compute RMSE, MAE, and R² for a fine-tuned SMILESTransformer.

    Params:
        model: SMILESTransformer : fine-tuned model in eval mode
        loader: DataLoader : LogDDataset DataLoader
        device: torch.device : inference device
    Returns:
        dict[str, float] : keys 'rmse', 'mae', 'r2'
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            out = (
                model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            preds.append(out)
            targets.append(batch["label"].numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def finetune(
    config: dict | None = None,
    checkpoint_dir: str | Path = "checkpoints/finetune",
) -> tuple[SMILESTransformer, dict[str, dict[str, float]]]:
    """
    Fine-tune a pretrained SMILESTransformer on the logD AstraZeneca dataset.

    Loads a phase-1 checkpoint (if provided), swaps in a LogDFinetuneHead, and
    fine-tunes end-to-end on the TDC scaffold split.

    Params:
        config: dict | None : hyperparameters; missing keys fall back to defaults
        checkpoint_dir: str | Path : directory for Lightning checkpoints
    Returns:
        tuple[SMILESTransformer, dict] : best fine-tuned model and metrics per split
    """
    cfg = {
        "pretrained_checkpoint": None,  # path to phase-1 .ckpt; None = no pretraining
        "d_hidden": 256,
        "dropout": 0.1,
        "lr": 1e-5,
        "weight_decay": 1e-2,
        "warmup_steps": 100,
        "batch_size": 32,
        "max_epochs": 50,
        "patience": 10,
        "max_length": 128,
        "seed": 42,
        "split": "tdc_scaffold",
        "wandb_project": "lipophilicity_pred",
        "wandb_run_name": None,
    }
    if config:
        cfg.update(config)

    L.seed_everything(cfg["seed"], workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    splits = get_tdc_split(seed=cfg["seed"])
    train_ds = LogDDataset(splits["train"], max_length=cfg["max_length"])
    val_ds = LogDDataset(splits["valid"], max_length=cfg["max_length"])
    test_ds = LogDDataset(splits["test"], max_length=cfg["max_length"])

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
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model ---
    head_kwargs = {
        "d_model": 768,
        "d_hidden": cfg["d_hidden"],
        "dropout": cfg["dropout"],
    }
    head = LogDFinetuneHead(**head_kwargs)
    pretrain_targets: list[str] = []

    pretrained_ckpt = cfg.get("pretrained_checkpoint")
    if pretrained_ckpt:
        model, pretrain_targets, _ = load_pretrained_backbone(pretrained_ckpt, device)
        model.swap_head(head)
    else:
        model = SMILESTransformer(head=head, model_name=CHEMBERTA_MODEL)
    model = model.to(device)

    # --- Loggers ---
    wandb_logger = WandbLogger(
        project=cfg["wandb_project"],
        name=cfg.get("wandb_run_name"),
        config={
            **cfg,
            "phase": "finetune",
            "pretrain_targets": pretrain_targets,
        },
    )
    csv_logger = CSVLogger(save_dir=str(ckpt_dir), name="logs")
    run_name = wandb_logger.experiment.name

    # --- Callbacks ---
    early_stop = EarlyStopping(monitor="val_mae", patience=cfg["patience"], mode="min")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name}-finetune-{{epoch:03d}}-{{val_mae:.4f}}",
        monitor="val_mae",
        mode="min",
        save_top_k=1,
    )

    lit = FinetuneLitModule(
        model,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        t_max=cfg["max_epochs"],
        head_kwargs=head_kwargs,
        pretrained_checkpoint=pretrained_ckpt,
        pretrain_targets=pretrain_targets,
    )

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="auto",
        callbacks=[early_stop, ckpt_cb],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=[wandb_logger, csv_logger],
    )
    trainer.fit(lit, train_loader, val_loader)

    import shutil

    history_src = Path(csv_logger.log_dir) / "metrics.csv"
    history_dst = ckpt_dir / "finetune_metrics_history.csv"
    if history_src.exists():
        shutil.copy(history_src, history_dst)

    # Load best weights back
    best_ckpt = torch.load(
        ckpt_cb.best_model_path, map_location=device, weights_only=False
    )
    model_state = {
        k[len("model.") :]: v
        for k, v in best_ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(model_state)
    best_model = model.to(device).eval()

    loaders = {"train": train_loader, "valid": val_loader, "test": test_loader}
    metrics = {
        split: evaluate_transformer(best_model, loader, device)
        for split, loader in loaders.items()
    }
    return best_model, metrics


def load_finetuned_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> SMILESTransformer:
    """
    Restore a fine-tuned SMILESTransformer from a Lightning checkpoint.

    Params:
        checkpoint_path: str | Path : path to the .ckpt file from finetune()
        device: torch.device | None : target device; defaults to CUDA if available
    Returns:
        SMILESTransformer : model in eval mode on the requested device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head_kwargs = ckpt.get(
        "head_kwargs", {"d_model": 768, "d_hidden": 256, "dropout": 0.1}
    )
    head = LogDFinetuneHead(**head_kwargs)
    model = SMILESTransformer(head=head, model_name=CHEMBERTA_MODEL)

    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state)
    return model.to(device).eval()
