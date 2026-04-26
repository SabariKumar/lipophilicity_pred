import traceback

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from src.utils import get_mol_fgs


def get_mol_descriptors(mol, missingVal=None) -> dict:
    """
    Compute the full list of RDKit descriptors for a single molecule.

    Params:
        mol: Chem.Mol : RDKit molecule object
        missingVal: any : value to use when a descriptor raises an exception
    Returns:
        dict : descriptor name → computed value (or missingVal on failure)
    """
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except Exception:
            traceback.print_exc()
            val = missingVal
        res[nm] = val
    return res


def smiles_to_descriptors(smiles: pd.Series) -> pd.DataFrame:
    """
    Compute RDKit descriptors for a Series of SMILES strings.

    Invalid SMILES produce a row of NaN rather than being dropped, preserving
    index alignment with the input for safe downstream merging.

    Params:
        smiles: pd.Series : SMILES strings, any index
    Returns:
        pd.DataFrame : shape (n, n_descriptors), index matches input, one column per RDKit descriptor
    """
    descriptor_names = [nm for nm, _ in Descriptors._descList]

    records = []
    for smi in tqdm(smiles, desc="Computing RDKit descriptors"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append({nm: np.nan for nm in descriptor_names})
        else:
            records.append(get_mol_descriptors(mol))

    return pd.DataFrame(records, index=smiles.index)


def smiles_to_fgs(smiles: pd.Series) -> pd.DataFrame:
    """
    Compute binary functional group presence for a Series of SMILES strings.

    Invalid SMILES produce a row of NaN rather than being dropped, preserving
    index alignment with the input. Functional group definitions come from
    RDKit's built-in hierarchy, flattened to a single level.

    Params:
        smiles: pd.Series : SMILES strings, any index
    Returns:
        pd.DataFrame : shape (n, n_fgs), index matches input, values are 0 or 1
    """
    records = []
    for smi in tqdm(smiles, desc="Computing functional groups"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append(None)
        else:
            records.append(get_mol_fgs(mol))

    return pd.DataFrame(records, index=smiles.index)
