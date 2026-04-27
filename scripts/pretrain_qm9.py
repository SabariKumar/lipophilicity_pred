"""
Pretrain the SMILES transformer on QM9 multi-task regression (phase 1).

Usage
-----
    # all 12 QM9 targets (default)
    pixi run pretrain-transformer

    # ablation: electronic properties only
    pixi run pretrain-transformer --targets homo lumo gap mu

    # custom hyperparameters
    pixi run pretrain-transformer --lr 1e-5 --max-epochs 20 --batch-size 64
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.qm9_data import QM9_TARGETS


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for QM9 pretraining.

    Params:
        None
    Returns:
        argparse.ArgumentParser : configured parser
    """
    p = argparse.ArgumentParser(
        description="Pretrain SMILES transformer on QM9",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=QM9_TARGETS,
        metavar="TARGET",
        help=(
            f"QM9 targets to pretrain on (default: all {len(QM9_TARGETS)}). "
            f"Choices: {QM9_TARGETS}"
        ),
    )
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument(
        "--max-length", type=int, default=None, help="Tokenizer max sequence length"
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--qm9-split",
        choices=["stratified_scaffold", "random"],
        default=None,
        help="QM9 split strategy for pretraining (default: stratified_scaffold)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/pretrain"),
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p


def main() -> None:
    """
    Parse arguments and run QM9 pretraining.

    Params:
        None
    Returns:
        None
    """
    args = _build_parser().parse_args()

    cli_overrides = {
        "targets": args.targets,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "max_length": args.max_length,
        "seed": args.seed,
        "qm9_split": args.qm9_split,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    cfg = {k: v for k, v in cli_overrides.items() if v is not None}

    from src.pretrain_transformer import (  # deferred: avoids heavy imports on --help
        pretrain,
    )

    best_ckpt = pretrain(config=cfg, checkpoint_dir=args.checkpoint_dir)
    print(f"\nBest pretrain checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
