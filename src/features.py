import traceback

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from src.utils import get_mol_fgs


def get_mol_descriptors(mol, missingVal=None) -> dict:
    """Calculate the full list of RDKit descriptors for a molecule.

    missingVal is used if a descriptor cannot be calculated.
    """
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            traceback.print_exc()
            val = missingVal
        res[nm] = val
    return res


def smiles_to_descriptors(smiles: pd.Series) -> pd.DataFrame:
    """Compute RDKit descriptors for a Series of SMILES strings.

    Invalid SMILES produce rows of NaN. Returns a DataFrame aligned to
    the input index with one column per RDKit descriptor.
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
    """Compute binary functional group presence for a Series of SMILES strings.

    Invalid SMILES produce rows of NaN. Returns a DataFrame aligned to the
    input index with one column per functional group (1 = present, 0 = absent).
    """
    records = []
    for smi in tqdm(smiles, desc="Computing functional groups"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append(None)
        else:
            records.append(get_mol_fgs(mol))

    return pd.DataFrame(records, index=smiles.index)
