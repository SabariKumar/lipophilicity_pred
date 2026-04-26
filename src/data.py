import pandas as pd
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


def get_splits(seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Return a scaffold split of the dataset as a dict with keys train/valid/test.

    Scaffold splitting groups structurally similar molecules into the same split,
    preventing the model from exploiting local structural memorisation across splits.

    Params:
        seed: int : random seed for reproducibility
    Returns:
        dict[str, pd.DataFrame] : each DataFrame has columns Drug_ID, Drug, Y
    """
    data = ADME(name="Lipophilicity_AstraZeneca")
    split = data.get_split(method="scaffold", seed=seed)
    return split
