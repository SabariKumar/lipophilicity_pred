import pandas as pd
from tdc.single_pred import ADME


def load_lipophilicity() -> pd.DataFrame:
    """Load Lipophilicity-AstraZeneca from TDC.

    Returns a DataFrame with columns:
        Drug_ID  - compound identifier
        Drug     - SMILES string
        Y        - logD (lipophilicity target)
    """
    data = ADME(name="Lipophilicity_AstraZeneca")
    return data.get_data()


def get_splits(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Return the TDC scaffold-split as a dict with keys train/valid/test."""
    data = ADME(name="Lipophilicity_AstraZeneca")
    split = data.get_split(method="scaffold", seed=seed)
    return split
