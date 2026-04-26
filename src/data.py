from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tdc.single_pred import ADME


def load_lipophilicity() -> pd.DataFrame:
    """
    Load the Lipophilicity-AstraZeneca dataset from TDC.

    Params:
        None
    Returns:
        pd.DataFrame : columns Drug_ID (str), Drug (SMILES), Y (logD float)
    """
    data = ADME(name="Lipophilicity_AstraZeneca")
    return data.get_data()


def _murcko_scaffold(smiles: str) -> str:
    """
    Return the canonical SMILES of the generic Murcko scaffold for a molecule.

    Ring-free molecules (e.g. linear aliphatics) return an empty string and are
    pooled into a single anonymous scaffold group.

    Params:
        smiles: str : SMILES string of the molecule
    Returns:
        str : canonical scaffold SMILES, or '' for ring-free molecules
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold_mol = GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold_mol) if scaffold_mol.GetNumAtoms() > 0 else ""


def get_tdc_split(seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Return the original TDC Murcko scaffold split for Lipophilicity-AstraZeneca.

    Scaffold groups are assigned to splits without balancing the logD
    distribution, so train/valid/test means can differ noticeably.  Use this
    as the uncontrolled scaffold-split baseline.

    Params:
        seed: int : random seed passed to TDC's scaffold splitter
    Returns:
        dict[str, pd.DataFrame] : keys 'train', 'valid', 'test'; each DataFrame
            has columns Drug_ID, Drug, Y with index reset
    """
    splits = ADME(name="Lipophilicity_AstraZeneca").get_split(
        method="scaffold", seed=seed
    )
    return {name: df.reset_index(drop=True) for name, df in splits.items()}


def get_random_split(
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """
    Return a purely random train/valid/test split with no scaffold awareness.

    Molecules are assigned to splits by random shuffling alone, so structurally
    similar compounds may appear in different splits.  Use this as a baseline
    to contrast against scaffold-based splits.

    Params:
        seed: int : random seed for reproducibility
        train_frac: float : fraction of molecules to allocate to training
        val_frac: float : fraction of molecules to allocate to validation
    Returns:
        dict[str, pd.DataFrame] : keys 'train', 'valid', 'test'; each DataFrame
            has columns Drug_ID, Drug, Y with index reset
    """
    from sklearn.model_selection import train_test_split

    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0, got {train_frac + val_frac}"
        )

    df = load_lipophilicity()
    train_val, test = train_test_split(df, test_size=test_frac, random_state=seed)
    train, valid = train_test_split(
        train_val,
        test_size=val_frac / (train_frac + val_frac),
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def get_splits(
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    n_strata: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Return a stratified Murcko scaffold split of the dataset.

    Molecules are grouped by their generic Murcko scaffold so that structurally
    similar molecules always land in the same split.  Scaffold groups are then
    binned into n_strata quantile strata by median logD, and a stratified
    shuffle split is applied at the scaffold-group level so that each split
    receives a representative slice of the logD distribution.

    The test fraction is inferred as 1 - train_frac - val_frac.  Splitting is
    performed in two phases using sklearn StratifiedShuffleSplit: first
    train vs (valid + test), then (valid + test) split equally into valid and
    test, both at the scaffold-group level.

    Params:
        seed: int : random seed for reproducibility
        train_frac: float : fraction of molecules to allocate to training
        val_frac: float : fraction of molecules to allocate to validation
        n_strata: int : number of quantile bins for logD stratification
    Returns:
        dict[str, pd.DataFrame] : keys 'train', 'valid', 'test'; each DataFrame
            has columns Drug_ID, Drug, Y with index reset
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0, got {train_frac + val_frac}"
        )

    df = load_lipophilicity().copy()
    df["_scaffold"] = df["Drug"].map(_murcko_scaffold)

    # One row per unique scaffold: median logD and molecule count.
    scaffold_stats = (
        df.groupby("_scaffold")["Y"].agg(median="median", n="count").reset_index()
    )

    # Warn if any single scaffold dominates the dataset.
    large_mask = scaffold_stats["n"] > 0.05 * len(df)
    for _, row in scaffold_stats[large_mask].iterrows():
        warnings.warn(
            f"Scaffold '{row['_scaffold'][:60]}' contains {row['n']} molecules "
            f"({100 * row['n'] / len(df):.1f}% of dataset) and will be assigned "
            f"entirely to one split.",
            stacklevel=2,
        )

    # Assign strata by rank so each stratum has ~equal scaffold-group count.
    # Using rank-based bins (not pd.qcut) avoids empty strata when many groups
    # share the same median logD value.
    scaffold_stats = scaffold_stats.sort_values("median").reset_index(drop=True)
    scaffold_stats["_stratum"] = (
        np.arange(len(scaffold_stats)) * n_strata // len(scaffold_stats)
    ).astype(int)

    # Phase 1: split scaffold groups into train vs (valid + test).
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_frac + test_frac), random_state=seed
    )
    train_idx, temp_idx = next(sss1.split(scaffold_stats, scaffold_stats["_stratum"]))

    # Phase 2: split (valid + test) scaffold groups 50/50 → valid and test.
    temp = scaffold_stats.iloc[temp_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
    )
    val_idx, test_idx = next(sss2.split(temp, temp["_stratum"]))
    valid_scaffolds = set(temp.iloc[val_idx]["_scaffold"])
    test_scaffolds = set(temp.iloc[test_idx]["_scaffold"])

    # Map every molecule to its split via its scaffold.
    def _assign(scaffold: str) -> str:
        if scaffold in valid_scaffolds:
            return "valid"
        if scaffold in test_scaffolds:
            return "test"
        return "train"

    df["_split"] = df["_scaffold"].map(_assign)
    df = df.drop(columns=["_scaffold"])

    return {
        split: df[df["_split"] == split].drop(columns=["_split"]).reset_index(drop=True)
        for split in ("train", "valid", "test")
    }
