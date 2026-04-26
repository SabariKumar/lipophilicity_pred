from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from chemprop.data import build_dataloader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import get_splits
from src.gnn_model import LipophilicityGNN
from src.graph_data import ChemBertaEncoder, build_chemprop_dataset

# ---------------------------------------------------------------------------
# Metric helper (mirrors evaluate() in the baseline models branch)
# ---------------------------------------------------------------------------


def evaluate_gnn(
    model: LipophilicityGNN,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """
    Compute RMSE, MAE, and R² for a fitted GNN on a chemprop DataLoader.

    Params:
        model: LipophilicityGNN : fitted model in eval mode
        loader: DataLoader : chemprop DataLoader yielding TrainingBatch tuples
        device: torch.device : device to run inference on
    Returns:
        dict[str, float] : keys 'rmse', 'mae', 'r2'
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch.bmg.to(device)  # in-place; BatchMolGraph.to() has no return value
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            Y = batch.Y.squeeze(-1)
            out = model(batch.bmg, X_d).squeeze(-1).cpu()
            preds.append(out.numpy())
            targets.append(Y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class GNNLitModule(L.LightningModule):
    """PyTorch Lightning wrapper for LipophilicityGNN."""

    def __init__(
        self,
        model: LipophilicityGNN,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 100,
        model_kwargs: dict | None = None,
    ) -> None:
        """
        Wrap a LipophilicityGNN for Lightning training.

        Params:
            model: LipophilicityGNN : the GNN model to train
            lr: float : AdamW learning rate
            weight_decay: float : AdamW weight decay
            t_max: int : CosineAnnealingLR period in epochs
            model_kwargs: dict | None : LipophilicityGNN constructor kwargs, embedded
                in the checkpoint so load_checkpoint can reconstruct the architecture
        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.loss_fn = nn.MSELoss()
        self._model_kwargs: dict = model_kwargs or {}

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Embed model architecture kwargs in the checkpoint for self-contained loading.

        Params:
            checkpoint: dict : Lightning checkpoint dict (mutated in place)
        Returns:
            None
        """
        checkpoint["model_kwargs"] = self._model_kwargs

    def _step(self, batch, split: str) -> torch.Tensor:
        bmg = batch.bmg
        X_d = batch.X_d
        Y = batch.Y.float()
        preds = self.model(bmg, X_d).squeeze(-1)
        targets = Y.squeeze(-1)
        loss = self.loss_fn(preds, targets)
        mae = (preds - targets).abs().mean()
        self.log(f"{split}_loss", loss, prog_bar=(split == "val"), batch_size=len(Y))
        self.log(f"{split}_mae", mae, prog_bar=(split == "val"), batch_size=len(Y))
        return loss

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """
        Compute MSE loss for a training batch.

        Params:
            batch: Any : chemprop TrainingBatch
            _batch_idx: int : batch index (unused)
        Returns:
            torch.Tensor : scalar MSE loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Any, _batch_idx: int) -> None:
        """
        Log validation MAE and loss without returning a value.

        Params:
            batch: Any : chemprop TrainingBatch
            _batch_idx: int : batch index (unused)
        Returns:
            None
        """
        self._step(batch, "val")

    def test_step(self, batch: Any, _batch_idx: int) -> None:
        """
        Log test MAE and loss without returning a value.

        Params:
            batch: Any : chemprop TrainingBatch
            _batch_idx: int : batch index (unused)
        Returns:
            None
        """
        self._step(batch, "test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Return AdamW optimizer with cosine annealing LR scheduler.

        Params:
            None
        Returns:
            dict : Lightning optimizer/scheduler config
        """
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_gnn(
    config: dict | None = None,
    checkpoint_dir: str | Path = "checkpoints",
) -> tuple[LipophilicityGNN, dict[str, dict[str, float]]]:
    """
    Full training pipeline: load data, encode ChemBERTa, train GNN, return metrics.

    Params:
        config: dict | None : hyperparameters; missing keys fall back to defaults
        checkpoint_dir: str | Path : directory for Lightning checkpoints
    Returns:
        tuple[LipophilicityGNN, dict] : best model and metrics per split
    """
    cfg = {
        "d_h": 300,
        "d_lm": 768,
        "d_hidden": 256,
        "depth": 3,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 20,
        "checkpoint_path": None,
        "seed": 42,
    }
    if config:
        cfg.update(config)

    L.seed_everything(cfg["seed"], workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    splits = get_splits(seed=cfg["seed"])
    encoder = ChemBertaEncoder().to(device)

    lm_embs = {
        name: encoder.encode(split["Drug"].tolist()) for name, split in splits.items()
    }
    datasets = {
        name: build_chemprop_dataset(
            splits[name]["Drug"],
            np.asarray(splits[name]["Y"].values, dtype=np.float32),
            lm_embs[name],
        )
        for name in ("train", "valid", "test")
    }
    loaders = {
        "train": build_dataloader(
            datasets["train"], batch_size=cfg["batch_size"], shuffle=True
        ),
        "valid": build_dataloader(
            datasets["valid"], batch_size=cfg["batch_size"], shuffle=False
        ),
        "test": build_dataloader(
            datasets["test"], batch_size=cfg["batch_size"], shuffle=False
        ),
    }

    # --- Model ---
    model_kwargs = {
        "d_h": cfg["d_h"],
        "d_lm": cfg["d_lm"],
        "d_hidden": cfg["d_hidden"],
        "depth": cfg["depth"],
        "dropout": cfg["dropout"],
        "checkpoint_path": cfg["checkpoint_path"],
    }
    model = LipophilicityGNN(**model_kwargs)
    lit = GNNLitModule(
        model,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        t_max=cfg["max_epochs"],
        model_kwargs=model_kwargs,
    )

    # --- Loggers ---
    wandb_logger = WandbLogger(
        project=cfg.get("wandb_project", "lipophilicity_pred"),
        name=cfg.get("wandb_run_name", None),
        config=cfg,  # logs all hyperparams including dropout, weight_decay, lr, etc.
    )
    csv_logger = CSVLogger(save_dir=str(ckpt_dir), name="logs")

    # Trigger wandb lazy init now so we can read the (possibly auto-generated) run name
    # and embed it in the checkpoint filename for traceability.
    run_name = wandb_logger.experiment.name

    # --- Callbacks ---
    early_stop = EarlyStopping(monitor="val_mae", patience=cfg["patience"], mode="min")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name}-{{epoch:03d}}-{{val_mae:.4f}}",
        monitor="val_mae",
        mode="min",
        save_top_k=1,
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
    trainer.fit(lit, loaders["train"], loaders["valid"])

    # Copy the per-epoch CSV to a stable path so the notebook can always find it.
    import shutil

    history_src = Path(csv_logger.log_dir) / "metrics.csv"
    history_dst = ckpt_dir / "metrics_history.csv"
    if history_src.exists():
        shutil.copy(history_src, history_dst)

    # Load best weights back into the unwrapped model.
    # Manually strip the "model." prefix that Lightning adds to nested module keys.
    ckpt_state = torch.load(ckpt_cb.best_model_path, map_location=device)
    model_state = {
        k[len("model.") :]: v
        for k, v in ckpt_state["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(model_state)
    best_model = model.to(device).eval()

    metrics = {
        split: evaluate_gnn(best_model, loaders[split], device)
        for split in ("train", "valid", "test")
    }
    return best_model, metrics


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    **model_kwargs,
) -> LipophilicityGNN:
    """
    Restore a LipophilicityGNN from a Lightning checkpoint written by train_gnn.

    Architecture hyperparameters are read from the checkpoint's embedded
    model_kwargs (written by GNNLitModule.on_save_checkpoint). Any kwargs
    passed by the caller override the checkpoint values, which is useful for
    the rare case of intentional architecture surgery but should not be needed
    in normal use.

    Params:
        checkpoint_path: str | Path : path to the .ckpt file
        device: torch.device | None : target device; defaults to CUDA if available
        **model_kwargs: override specific LipophilicityGNN constructor kwargs;
            normally empty — the checkpoint is self-contained
    Returns:
        LipophilicityGNN : model in eval mode on the requested device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = {
        k[len("model.") :]: v
        for k, v in ckpt_state["state_dict"].items()
        if k.startswith("model.")
    }

    # Prefer kwargs embedded by on_save_checkpoint; fall back to inferring
    # d_h / d_lm / d_hidden from weight shapes for checkpoints that pre-date
    # this feature.  depth and dropout don't affect tensor shapes so they stay
    # at their defaults when not stored.
    if "model_kwargs" in ckpt_state:
        kwargs = {**ckpt_state["model_kwargs"], **model_kwargs}
    else:
        d_h = model_state["backbone.mp.W_o.bias"].shape[0]
        d_lm = model_state["fusion.net.0.weight"].shape[0] - d_h
        d_hidden = model_state["fusion.net.1.bias"].shape[0]
        kwargs = {"d_h": d_h, "d_lm": d_lm, "d_hidden": d_hidden, **model_kwargs}

    model = LipophilicityGNN(**kwargs)
    model.load_state_dict(model_state)
    return model.to(device).eval()
