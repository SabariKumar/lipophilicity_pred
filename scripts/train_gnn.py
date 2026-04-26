"""
Train the GNN + ChemBERTa lipophilicity model.

Usage
-----
    # defaults
    pixi run python scripts/train_gnn.py

    # common overrides
    pixi run python scripts/train_gnn.py --lr 5e-4 --max-epochs 200 --batch-size 32

    # load from a YAML config (individual flags override the file)
    pixi run python scripts/train_gnn.py --config configs/gnn_base.yaml --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the training script.

    Params:
        None
    Returns:
        argparse.ArgumentParser : configured parser with all training flags
    """
    p = argparse.ArgumentParser(
        description="Train GNN + ChemBERTa lipophilicity model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file; CLI flags override file values",
    )
    # Model architecture
    p.add_argument("--d-h", type=int, default=None, help="MPNN hidden dim")
    p.add_argument("--d-hidden", type=int, default=None, help="FusionMLP hidden dim")
    p.add_argument("--depth", type=int, default=None, help="message-passing steps")
    p.add_argument("--dropout", type=float, default=None)
    # Optimisation
    p.add_argument("--lr", type=float, default=None, help="AdamW learning rate")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument(
        "--patience", type=int, default=None, help="early-stopping patience (epochs)"
    )
    # Pretrained GNN checkpoint to initialise the backbone from
    p.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
        help="chemprop .ckpt to load backbone weights from",
    )
    # Misc
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="directory for Lightning model checkpoints",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["stratified_scaffold", "random", "tdc_scaffold"],
        help="train/valid/test split strategy (default: stratified_scaffold)",
    )
    # Wandb
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (default: lipophilicity_pred)",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name; auto-generated if omitted",
    )
    return p


def main() -> None:
    """
    Parse arguments, run train_gnn, and print a metrics summary.

    Params:
        None
    Returns:
        None
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Start from YAML base config if provided, then apply CLI overrides.
    cfg: dict = {}
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    cli_overrides = {
        "d_h": args.d_h,
        "d_hidden": args.d_hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "checkpoint_path": args.backbone_checkpoint,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "split": args.split,
    }
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})

    from src.train_gnn import (  # deferred so --help works without heavy imports
        train_gnn,
    )

    _, metrics = train_gnn(config=cfg, checkpoint_dir=args.checkpoint_dir)

    # Print a clean metrics table to stdout.
    print("\n" + "=" * 46)
    print(f"{'split':<8}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}")
    print("-" * 46)
    for split, m in metrics.items():
        print(f"{split:<8}  {m['rmse']:>7.4f}  {m['mae']:>7.4f}  {m['r2']:>7.4f}")
    print("=" * 46)

    # Also write metrics to a JSON file beside the checkpoint dir for easy parsing.
    out_path = Path(args.checkpoint_dir) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
