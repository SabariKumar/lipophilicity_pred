from collections import namedtuple
from typing import Dict, List, Tuple

import rdkit
from rdkit.Chem import FunctionalGroups

_Pattern = namedtuple("pattern", "smarts mol")


def _flatten_fgs(fgs: List) -> Tuple[List, Dict]:
    """
    Recursively flatten rdkit's functional group hierarchy into a single dict.

    Params:
        fgs: List : Nested list of fgs from rdkit's FunctionalGroups.BuildFuncGroupHierarchy()
    Returns:
        Tuple[List, Dict] : sorted fg names, flattened fg definitions
    """
    res = {}
    if not fgs:
        return [], res
    for x in fgs:
        res[x.label] = _Pattern(x.smarts, x.pattern)
        _, child_defs = _flatten_fgs(x.children)
        res.update(child_defs)
    return sorted(res.keys()), res


_FG_NAMES: List[str] | None = None
_FG_DEFS: Dict | None = None


def _get_fg_library() -> Tuple[List[str], Dict]:
    """
    Return the cached flattened functional group library.

    Params:
        None
    Returns:
        Tuple[List[str], Dict] : fg names, fg definitions
    """
    global _FG_NAMES, _FG_DEFS
    if _FG_NAMES is None:
        fgh = FunctionalGroups.BuildFuncGroupHierarchy()
        _FG_NAMES, _FG_DEFS = _flatten_fgs(fgh)
    return _FG_NAMES, _FG_DEFS


def get_mol_fgs(mol: rdkit.Chem.rdchem.Mol) -> Dict[str, int]:
    """
    Return binary functional group presence for a single molecule.

    Params:
        mol: rdkit.Chem.rdchem.Mol : input rdkit mol object
    Returns:
        Dict[str, int] : mapping of fg name to 1 (present) or 0 (absent)
    """
    all_fg_names, all_fg_defs = _get_fg_library()
    return {
        fgn: int(mol.HasSubstructMatch(all_fg_defs[fgn].mol)) for fgn in all_fg_names
    }
