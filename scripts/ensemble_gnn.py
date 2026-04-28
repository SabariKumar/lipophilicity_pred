"""
Train an ensemble of LipophilicityGNN models and aggregate predictions.

Each ensemble member is an independent run of train_gnn with a different
random seed. After training, all members are loaded and run on the same
test split to produce per-molecule mean and std predictions.

Usage
-----
    # train 5 members then aggregate (default seeds 42–46)
    pixi run ensemble-gnn

    # explicit seeds
    pixi run ensemble-gnn --seeds 0 1 2 3 4

    # skip training (members already trained), just aggregate
    pixi run ensemble-gnn --aggregate-only --checkpoint-dir checkpoints/uq/ensemble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the ensemble training script.

    Params:
        None
    Returns:
        argparse.ArgumentParser : configured parser
    """
    p = argparse.ArgumentParser(
        description="Train and aggregate a LipophilicityGNN ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="number of ensemble members (ignored if --seeds is given)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="explicit seed list, e.g. --seeds 0 1 2 3 4",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/uq/ensemble"),
        help="root directory; each seed gets a subdirectory seed_<N>",
    )
    p.add_argument(
        "--split",
        type=str,
        default="tdc_scaffold",
        choices=["stratified_scaffold", "random", "tdc_scaffold"],
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--wandb-project", type=str, default="lipophilicity_pred")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip training; load existing checkpoints and aggregate",
    )
    return p


def _find_checkpoint(seed_dir: Path) -> Path:
    """
    Return the single .ckpt file written by ModelCheckpoint in seed_dir.

    Params:
        seed_dir: Path : directory for one ensemble member
    Returns:
        Path : path to the checkpoint file
    """
    ckpts = [p for p in seed_dir.glob("*.ckpt") if "last" not in p.name]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {seed_dir}")
    # ModelCheckpoint(save_top_k=1) writes exactly one file; sort as tiebreak.
    return sorted(ckpts)[0]


def _build_test_loader(split: str, batch_size: int, device):
    """
    Build chemprop DataLoaders for train, valid, and test splits.

    Params:
        split: str : split strategy ('tdc_scaffold', 'stratified_scaffold', 'random')
        batch_size: int : loader batch size
        device: torch.device : device for ChemBertaEncoder
    Returns:
        tuple[dict, dict] : (splits dict of DataFrames, loaders dict of DataLoaders)
    """
    import numpy as np
    from chemprop.data import build_dataloader

    from src.data import get_random_split, get_splits, get_tdc_split
    from src.graph_data import ChemBertaEncoder, build_chemprop_dataset

    _split_fns = {
        "stratified_scaffold": get_splits,
        "random": get_random_split,
        "tdc_scaffold": get_tdc_split,
    }
    splits = _split_fns[split](seed=42)
    encoder = ChemBertaEncoder().to(device)
    lm_embs = {name: encoder.encode(df["Drug"].tolist()) for name, df in splits.items()}
    datasets = {
        name: build_chemprop_dataset(
            splits[name]["Drug"],
            np.asarray(splits[name]["Y"].values, dtype=np.float32),
            lm_embs[name],
        )
        for name in ("train", "valid", "test")
    }
    loaders = {
        name: build_dataloader(datasets[name], batch_size=batch_size, shuffle=False)
        for name in ("train", "valid", "test")
    }
    return splits, loaders


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return predictions and targets.

    Params:
        model: LipophilicityGNN : model in eval mode
        loader: DataLoader : chemprop DataLoader
        device: torch.device : inference device
    Returns:
        tuple[ndarray, ndarray] : (predictions (N,), targets (N,))
    """
    import torch

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch.bmg.to(device)
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            out = model(batch.bmg, X_d).squeeze(-1).cpu().numpy()
            preds.append(out)
            targets.append(batch.Y.squeeze(-1).numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    """
    Train ensemble members, aggregate predictions, and save results.

    Params:
        None
    Returns:
        None
    """
    import pandas as pd
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from src.train_gnn import load_checkpoint, train_gnn

    args = _build_parser().parse_args()
    seeds = args.seeds if args.seeds is not None else list(range(42, 42 + args.n_seeds))
    ckpt_root = args.checkpoint_dir
    ckpt_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Phase 1: train ---
    if not args.aggregate_only:
        base_cfg = {
            "split": args.split,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "wandb_project": args.wandb_project,
        }
        for seed in seeds:
            seed_dir = ckpt_root / f"seed_{seed}"
            print(f"\n{'='*60}\nTraining ensemble member seed={seed}\n{'='*60}")
            cfg = {**base_cfg, "seed": seed, "wandb_run_name": f"ensemble-seed-{seed}"}
            _, metrics = train_gnn(config=cfg, checkpoint_dir=seed_dir)
            print(
                f"seed={seed}  test RMSE={metrics['test']['rmse']:.4f}  "
                f"MAE={metrics['test']['mae']:.4f}  R²={metrics['test']['r2']:.4f}"
            )
            with open(seed_dir / "metrics.json", "w") as f:
                json.dump({"seed": seed, **metrics}, f, indent=2)

    # --- Phase 2: aggregate ---
    print(f"\n{'='*60}\nAggregating ensemble predictions\n{'='*60}")
    splits_df, loaders = _build_test_loader(args.split, args.batch_size, device)

    all_preds: dict[str, list[np.ndarray]] = {"train": [], "valid": [], "test": []}
    y_true: dict[str, np.ndarray] = {}

    for seed in seeds:
        seed_dir = ckpt_root / f"seed_{seed}"
        ckpt_path = _find_checkpoint(seed_dir)
        print(f"Loading {ckpt_path.name} (seed={seed})")
        model = load_checkpoint(ckpt_path, device=device)
        for split_name in ("train", "valid", "test"):
            preds, targets = _predict(model, loaders[split_name], device)
            all_preds[split_name].append(preds)
            if split_name not in y_true:
                y_true[split_name] = targets

    print("\nEnsemble metrics:")
    print(f"{'split':<8}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'mean σ':>8}")
    print("-" * 50)
    ensemble_results: dict[str, dict] = {}
    for split_name in ("train", "valid", "test"):
        stacked = np.stack(all_preds[split_name], axis=0)  # (N_seeds, N_mols)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        yt = y_true[split_name]
        rmse = float(mean_squared_error(yt, mean) ** 0.5)
        mae = float(mean_absolute_error(yt, mean))
        r2 = float(r2_score(yt, mean))
        print(
            f"{split_name:<8}  {rmse:>7.4f}  {mae:>7.4f}  {r2:>7.4f}  {std.mean():>8.4f}"
        )
        ensemble_results[split_name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mean_std": float(std.mean()),
        }

    # Save test-split predictions CSV
    test_stacked = np.stack(all_preds["test"], axis=0)
    test_mean = test_stacked.mean(axis=0)
    test_std = test_stacked.std(axis=0)
    out_df = pd.DataFrame(
        {
            "smiles": splits_df["test"]["Drug"].tolist(),
            "y_true": y_true["test"],
            "mean": test_mean,
            "std": test_std,
            **{f"pred_seed_{s}": test_stacked[i] for i, s in enumerate(seeds)},
        }
    )
    csv_path = ckpt_root / "ensemble_predictions.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\nTest predictions saved to {csv_path}")

    metrics_path = ckpt_root / "ensemble_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"seeds": seeds, **ensemble_results}, f, indent=2)
    print(f"Ensemble metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
