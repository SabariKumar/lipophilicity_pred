from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# TDC label names for QM9 (retrieved via retrieve_label_name_list('QM9')).
# Rotational constants A, B, C are excluded — they carry little signal for
# property prediction tasks and are not used in standard QM9 ML benchmarks.
QM9_TARGETS = [
    "mu",  # dipole moment (D)
    "alpha",  # polarisability (a₀³)
    "homo",  # HOMO energy (Ha)
    "lumo",  # LUMO energy (Ha)
    "gap",  # HOMO-LUMO gap (Ha)
    "r2",  # electronic spatial extent (a₀²)
    "zpve",  # zero-point vibrational energy (Ha)
    "U0",  # internal energy at 0 K (Ha)
    "U",  # internal energy at 298 K (Ha)
    "H",  # enthalpy at 298 K (Ha)
    "G",  # free energy at 298 K (Ha)
    "Cv",  # heat capacity at 298 K (cal/mol/K)
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_QM9_PKL = _PROJECT_ROOT / "data" / "qm9.pkl"
_QM9_CACHE = _PROJECT_ROOT / "data" / "qm9_smiles.parquet"


def _patch_pandas_compat() -> None:
    """
    Patch pandas 2.x to load DataFrames pickled with pandas 1.x.

    TDC caches data as pickle files created with pandas 1.x, where
    BlockPlacement could be stored as a plain slice.  Pandas 2.x requires
    BlockPlacement objects, causing a TypeError on load.  This patch
    intercepts new_block calls and converts slice placements in-place.
    The patch is applied at most once per process.

    Params:
        None
    Returns:
        None
    """
    import pandas.core.internals.blocks as _blocks
    from pandas._libs.internals import BlockPlacement

    if getattr(_blocks, "_compat_patched", False):
        return

    _orig = _blocks.new_block

    def _patched(values, placement, ndim=2, refs=None):
        if isinstance(placement, slice):
            n = (
                placement.stop
                if placement.stop is not None
                else (values.shape[-1] if hasattr(values, "shape") else len(values))
            )
            placement = BlockPlacement(range(*placement.indices(n)))
        return _orig(values, placement, ndim=ndim, refs=refs)

    _blocks.new_block = _patched
    _blocks._compat_patched = True


def _xyz_to_smiles(x: tuple) -> str | None:
    """
    Convert a QM9 (atom_types, xyz_coords) tuple to canonical SMILES.

    Uses RDKit DetermineBonds to infer connectivity from 3D coordinates,
    then strips explicit hydrogens and returns canonical SMILES.  Returns
    None if bond determination or SMILES generation fails.  RDKit error
    logging is suppressed here because DetermineBonds on malformed
    structures produces high-volume valence warnings that are expected
    and already handled by the None return.

    Params:
        x: tuple : (list[str], ndarray) — atom element symbols and Å coords
    Returns:
        str | None : canonical SMILES, or None on failure
    """
    from rdkit import Chem
    from rdkit.Chem.rdDetermineBonds import DetermineBonds
    from rdkit.rdBase import DisableLog, EnableLog

    atoms, coords = x
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, (atom, xyz) in enumerate(zip(atoms, np.array(coords))):
        mol.AddAtom(Chem.Atom(atom))
        conf.SetAtomPosition(i, xyz.tolist())
    mol.AddConformer(conf)
    DisableLog("rdApp.error")
    try:
        DetermineBonds(mol)
        mol = Chem.RemoveHs(mol.GetMol())
        result = Chem.MolToSmiles(mol)
    except Exception:
        result = None
    finally:
        EnableLog("rdApp.error")
    return result


def _murcko_scaffold(smiles: str) -> str:
    """
    Return the canonical SMILES of the generic Murcko scaffold.

    Ring-free molecules (common in QM9 due to its small, simple structures)
    return an empty string and are pooled into a single anonymous group.

    Params:
        smiles: str : SMILES string of the molecule
    Returns:
        str : canonical scaffold SMILES, or '' for ring-free molecules
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold_mol = GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold_mol) if scaffold_mol.GetNumAtoms() > 0 else ""


def _build_smiles_df() -> pd.DataFrame:
    """
    Load the raw QM9 pickle, convert 3D structures to SMILES, and drop failures.

    TDC's QM9 stores molecules as (atom_types, xyz_coords) tuples.  TDC's own
    multi-label API is broken for list label_name inputs (fuzzy_search calls
    name.lower() on a list), so we load the raw pickle directly and use RDKit
    DetermineBonds for SMILES generation.

    Params:
        None
    Returns:
        pd.DataFrame : columns Drug (SMILES), Drug_ID, and 12 QM9 targets
    """
    _patch_pandas_compat()
    raw = pd.read_pickle(_QM9_PKL)
    raw["Drug"] = raw["X"].apply(_xyz_to_smiles)
    raw["Drug_ID"] = raw["ID"]
    keep_cols = ["Drug", "Drug_ID"] + QM9_TARGETS
    df = raw[keep_cols].dropna(subset=["Drug"] + QM9_TARGETS).reset_index(drop=True)
    return df


def load_qm9() -> pd.DataFrame:
    """
    Load the QM9 dataset with SMILES strings and 12 standard QM targets.

    On first call, the raw TDC pickle is loaded, 3D structures are converted to
    SMILES via RDKit DetermineBonds, and the result is cached as parquet in
    data/qm9_smiles.parquet for fast subsequent loads.  Molecules where SMILES
    conversion fails (~1%) are silently dropped.

    Params:
        None
    Returns:
        pd.DataFrame : ~133k rows; columns Drug (SMILES), Drug_ID, + 12 targets
    """
    if _QM9_CACHE.exists():
        return pd.read_parquet(_QM9_CACHE)
    print("Building QM9 SMILES cache (one-time, ~30s)...")
    df = _build_smiles_df()
    _QM9_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_QM9_CACHE, index=False)
    print(f"Cached {len(df):,} molecules to {_QM9_CACHE}")
    return df


def get_qm9_splits(
    seed: int = 42,
    method: str = "stratified_scaffold",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    n_strata: int = 5,
    stratify_on: str = "gap",
) -> dict[str, pd.DataFrame]:
    """
    Return a train / valid / test split of QM9.

    Two split strategies are supported:

    ``"random"``
        Plain stratified random split — molecules are assigned purely by
        shuffling.  Structurally similar molecules may appear across splits.

    ``"stratified_scaffold"``
        Molecules are first grouped by their generic Murcko scaffold so
        structurally similar molecules always land in the same split.  Scaffold
        groups are then binned into ``n_strata`` quantile strata by median value
        of ``stratify_on`` (default: HOMO-LUMO gap), and a stratified shuffle
        split is applied at the scaffold-group level.  This ensures each split
        sees a representative cross-section of the QM9 property distribution
        while preserving scaffold integrity.  QM9 contains many ring-free
        molecules (linear aliphatics); these are pooled into a single
        scaffold group.

    Params:
        seed: int : random seed for reproducibility
        method: str : 'random' or 'stratified_scaffold'
        train_frac: float : fraction of molecules for training
        val_frac: float : fraction of molecules for validation
        n_strata: int : number of quantile bins for property stratification
        stratify_on: str : QM9 target column used to stratify scaffold groups
    Returns:
        dict[str, pd.DataFrame] : keys 'train', 'valid', 'test'
    """
    if method == "random":
        return _random_split(seed=seed, train_frac=train_frac, val_frac=val_frac)
    if method == "stratified_scaffold":
        return _stratified_scaffold_split(
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            n_strata=n_strata,
            stratify_on=stratify_on,
        )
    raise ValueError(
        f"Unknown method '{method}'. Use 'random' or 'stratified_scaffold'."
    )


def _random_split(
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, pd.DataFrame]:
    """
    Return a purely random 80/10/10 split of QM9.

    Params:
        seed: int : random seed
        train_frac: float : training fraction
        val_frac: float : validation fraction
    Returns:
        dict[str, pd.DataFrame] : keys 'train', 'valid', 'test'
    """
    from sklearn.model_selection import train_test_split

    test_frac = 1.0 - train_frac - val_frac
    df = load_qm9()
    train_val, test = train_test_split(df, test_size=test_frac, random_state=seed)
    train, valid = train_test_split(
        train_val, test_size=val_frac / (train_frac + val_frac), random_state=seed
    )
    return {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _stratified_scaffold_split(
    seed: int,
    train_frac: float,
    val_frac: float,
    n_strata: int,
    stratify_on: str,
) -> dict[str, pd.DataFrame]:
    """
    Return a stratified Murcko scaffold split of QM9.

    Scaffold groups are stratified by the median value of ``stratify_on``
    across molecules sharing that scaffold.  The split mirrors the approach
    used for the logD lipophilicity dataset in src/data.py.

    Params:
        seed: int : random seed
        train_frac: float : training fraction
        val_frac: float : validation fraction
        n_strata: int : number of quantile strata
        stratify_on: str : QM9 target column name for scaffold stratification
    Returns:
        dict[str, pd.DataFrame] : keys 'train', 'valid', 'test'
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    test_frac = 1.0 - train_frac - val_frac

    df = load_qm9().copy()
    df["_scaffold"] = df["Drug"].map(_murcko_scaffold)

    scaffold_stats = (
        df.groupby("_scaffold")[stratify_on]
        .agg(median="median", n="count")
        .reset_index()
    )

    large_mask = scaffold_stats["n"] > 0.05 * len(df)
    for _, row in scaffold_stats[large_mask].iterrows():
        warnings.warn(
            f"Scaffold '{row['_scaffold'][:60]}' contains {row['n']} molecules "
            f"({100 * row['n'] / len(df):.1f}% of dataset) and will be assigned "
            f"entirely to one split.",
            stacklevel=2,
        )

    scaffold_stats = scaffold_stats.sort_values("median").reset_index(drop=True)
    scaffold_stats["_stratum"] = (
        np.arange(len(scaffold_stats)) * n_strata // len(scaffold_stats)
    ).astype(int)

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_frac + test_frac), random_state=seed
    )
    _, temp_idx = next(sss1.split(scaffold_stats, scaffold_stats["_stratum"]))

    temp = scaffold_stats.iloc[temp_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
    )
    val_idx, test_idx = next(sss2.split(temp, temp["_stratum"]))
    valid_scaffolds = set(temp.iloc[val_idx]["_scaffold"])
    test_scaffolds = set(temp.iloc[test_idx]["_scaffold"])

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


class QM9Normalizer:
    """
    Per-target z-score normaliser fitted on the QM9 training split.

    QM9 targets span very different scales and units (eV, Bohr, kcal/mol).
    Normalising each target independently to zero mean and unit variance lets
    the multi-task loss treat all targets equally without hand-tuned weights.

    The normaliser must be fitted on the training split only and the fitted
    statistics are saved alongside the pretrained checkpoint so they can be
    used at inference time (e.g. to denormalise predictions for inspection).
    """

    def __init__(self) -> None:
        """
        Initialise an unfitted normaliser.

        Params:
            None
        Returns:
            None
        """
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, df: pd.DataFrame, targets: list[str]) -> "QM9Normalizer":
        """
        Compute per-target mean and std from a training DataFrame.

        Params:
            df: pd.DataFrame : training split containing target columns
            targets: list[str] : column names of the targets to normalise
        Returns:
            QM9Normalizer : self, for chaining
        """
        self.mean_ = df[targets].mean()
        self.std_ = df[targets].std().clip(lower=1e-8)
        return self

    def transform(self, df: pd.DataFrame, targets: list[str]) -> np.ndarray:
        """
        Apply z-score normalisation and return a float32 array.

        Params:
            df: pd.DataFrame : DataFrame containing target columns
            targets: list[str] : column names to normalise (must match fit)
        Returns:
            np.ndarray : shape (n, len(targets)), dtype float32
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        normed = (df[targets] - self.mean_[targets]) / self.std_[targets]
        return normed.values.astype(np.float32)

    def state_dict(self) -> dict:
        """
        Return serialisable mean/std arrays for checkpoint embedding.

        Params:
            None
        Returns:
            dict : keys 'mean' and 'std' as lists, plus 'targets'
        """
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "targets": self.mean_.index.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "QM9Normalizer":
        """
        Reconstruct a normaliser from a saved state dict.

        Params:
            state: dict : output of state_dict()
        Returns:
            QM9Normalizer : fitted normaliser
        """
        norm = cls()
        norm.mean_ = pd.Series(state["mean"], index=state["targets"])
        norm.std_ = pd.Series(state["std"], index=state["targets"])
        return norm
