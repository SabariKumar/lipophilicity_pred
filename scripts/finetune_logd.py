"""
Fine-tune the pretrained SMILES transformer on logD (phase 2).

Usage
-----
    # with pretrained backbone
    pixi run finetune-transformer --pretrained-checkpoint checkpoints/pretrain/<run>.ckpt

    # no-pretraining baseline (unfrozen ChemBERTa fine-tuned directly)
    pixi run finetune-transformer

    # custom hyperparameters
    pixi run finetune-transformer --pretrained-checkpoint <path> --lr 5e-6 --max-epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for logD fine-tuning.

    Params:
        None
    Returns:
        argparse.ArgumentParser : configured parser
    """
    p = argparse.ArgumentParser(
        description="Fine-tune SMILES transformer on logD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Phase-1 .ckpt path; omit to fine-tune without pretraining (baseline)",
    )
    p.add_argument(
        "--d-hidden", type=int, default=None, help="LogDFinetuneHead hidden dim"
    )
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/finetune"),
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p


def main() -> None:
    """
    Parse arguments, run fine-tuning, and print metrics summary.

    Params:
        None
    Returns:
        None
    """
    args = _build_parser().parse_args()

    cli_overrides = {
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "d_hidden": args.d_hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "max_length": args.max_length,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    cfg = {k: v for k, v in cli_overrides.items() if v is not None}

    from src.finetune_transformer import (  # deferred: avoids heavy imports on --help
        finetune,
    )

    _, metrics = finetune(config=cfg, checkpoint_dir=args.checkpoint_dir)

    print("\n" + "=" * 46)
    print(f"{'split':<8}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}")
    print("-" * 46)
    for split, m in metrics.items():
        print(f"{split:<8}  {m['rmse']:>7.4f}  {m['mae']:>7.4f}  {m['r2']:>7.4f}")
    print("=" * 46)

    out_path = Path(args.checkpoint_dir) / "finetune_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
