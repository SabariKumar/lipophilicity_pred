from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from src.gnn_model import LipophilicityGNN
from src.graph_data import build_chemprop_dataset


def _make_single_loader(smiles: str, lm_emb: np.ndarray):
    """
    Build a single-molecule chemprop DataLoader for inference and attribution.

    Params:
        smiles: str : SMILES string for the molecule
        lm_emb: np.ndarray : ChemBERTa [CLS] embedding, shape (768,)
    Returns:
        DataLoader : chemprop loader yielding one batch of size 1
    """
    from chemprop.data import build_dataloader

    dataset = build_chemprop_dataset(
        smiles=[smiles],
        targets=np.array([0.0]),
        lm_embeddings=lm_emb[np.newaxis],
    )
    return build_dataloader(dataset, batch_size=1, shuffle=False)


class AtomAttributionExplainer:
    """
    Computes per-atom attribution scores using Captum IntegratedGradients.

    IG is applied to the atom feature matrix (bmg.V) with a zero-vector baseline
    representing an absent atom. At each integration step, bmg.V is replaced with
    the alpha-scaled atom features so gradients flow through the full message-passing
    network. Scores are the L2 norm of the attribution tensor over the feature
    dimension, giving one non-negative scalar per heavy atom.

    This approach avoids the BatchMolGraph-incompatibility of LayerIntegratedGradients,
    which requires Captum to create baselines from non-tensor inputs.
    """

    def __init__(
        self, model: LipophilicityGNN, device: torch.device | None = None
    ) -> None:
        """
        Initialise IntegratedGradients targeting the atom feature matrix.

        Params:
            model: LipophilicityGNN : trained model (must be in eval mode)
            device: torch.device | None : inference device; defaults to model's device
        Returns:
            None
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

        def _fwd(V: torch.Tensor, bmg, X_d: torch.Tensor | None) -> torch.Tensor:
            # Patch atom features so IG can vary them along the integration path.
            bmg.V = V
            return model(bmg, X_d).squeeze(-1)

        self.ig = IntegratedGradients(_fwd)

    def explain(
        self,
        smiles: str,
        lm_emb: np.ndarray,
        n_steps: int = 50,
    ) -> np.ndarray:
        """
        Return per-atom attribution scores for a single molecule.

        IG integrates the gradient of the model output w.r.t. the atom feature
        matrix from a zero baseline to the actual features. Scores are the L2
        norm over the feature dimension, giving one non-negative value per atom.

        Params:
            smiles: str : SMILES string of the molecule to explain
            lm_emb: np.ndarray : ChemBERTa [CLS] embedding, shape (768,)
            n_steps: int : number of integration steps (higher = more accurate)
        Returns:
            np.ndarray : per-atom scores, shape (n_heavy_atoms,), float32
        """
        loader = _make_single_loader(smiles, lm_emb)
        batch = next(iter(loader))
        batch.bmg.to(self.device)  # in-place; returns None
        bmg = batch.bmg
        X_d = batch.X_d.to(self.device) if batch.X_d is not None else None

        V = bmg.V.detach().clone()
        attr = self.ig.attribute(
            inputs=V,
            baselines=torch.zeros_like(V),
            additional_forward_args=(bmg, X_d),
            n_steps=n_steps,
            # One alpha step per forward call so bmg.batch (fixed, single molecule)
            # stays consistent with the atom count in V.
            internal_batch_size=1,
        )
        # Sum over the feature dimension to preserve sign:
        # positive = atom features push logD up (hydrophobic, red)
        # negative = atom features pull logD down (hydrophilic, blue)
        # norm() would discard the sign and make everything non-negative.
        scores = attr.sum(dim=-1).detach().cpu().numpy().astype(np.float32)
        return scores


def plot_atom_contributions(
    smiles: str,
    scores: np.ndarray,
    title: str = "",
    size: tuple[int, int] = (400, 300),
) -> plt.Figure:
    """
    Draw a 2D molecule coloured by per-atom attribution scores.

    Atoms with positive logD contribution (hydrophobic) are coloured red;
    atoms with negative contribution (hydrophilic) are coloured blue.
    Scores are normalised to [-1, 1] across the molecule before mapping to
    colour, so the colour scale is consistent within one molecule but not
    comparable across different molecules.

    Params:
        smiles: str : SMILES string of the molecule
        scores: np.ndarray : per-atom attribution scores, shape (n_heavy_atoms,)
        title: str : optional figure title
        size: tuple[int, int] : pixel dimensions for the structure image
    Returns:
        plt.Figure : matplotlib figure containing the coloured structure
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    max_abs = float(np.abs(scores).max()) or 1.0
    norm_scores = scores / max_abs  # in [-1, 1]

    cmap = plt.get_cmap("RdBu_r")  # red = positive, blue = negative
    atom_colours: dict[int, tuple] = {}
    alpha = 0.5
    for i, s in enumerate(norm_scores):
        r, g, b, _ = cmap((s + 1.0) / 2.0)  # map [-1,1] → [0,1]
        atom_colours[i] = (r, g, b, alpha)

    import io

    from PIL import Image

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addAtomIndices = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colours,
        highlightBonds=[],
        highlightBondColors={},
    )
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Attribution score", fontsize=8)
    plt.tight_layout()
    return fig
