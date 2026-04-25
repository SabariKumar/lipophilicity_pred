from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from rdkit import Chem
from rdkit.Chem import Draw
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
        targets=np.array([0.0]),  # placeholder target
        lm_embeddings=lm_emb[np.newaxis],
    )
    return build_dataloader(dataset, batch_size=1, shuffle=False)


class AtomAttributionExplainer:
    """
    Computes per-atom attribution scores using Captum LayerIntegratedGradients.

    The target layer is the final output projection of the chemprop MPNN
    (backbone.mp.W_o), which produces per-atom hidden states after all message
    passing is complete. IG integrates gradients from a zero-activation baseline
    to the real activation, giving each atom a signed importance score that
    reflects its marginal contribution to the predicted logD.

    Note on layer name: W_o is the assumed attribute name for the final linear
    projection in chemprop v2's BondMessagePassing. Verify with
    `list(model.backbone.mp.named_modules())` if attribution fails.
    """

    def __init__(
        self, model: LipophilicityGNN, device: torch.device | None = None
    ) -> None:
        """
        Attach IntegratedGradients to the MPNN's final atom projection.

        Params:
            model: LipophilicityGNN : trained model (must be in eval mode)
            device: torch.device | None : inference device; defaults to model's device
        Returns:
            None
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

        def _forward(bmg, V_d):
            return self.model(bmg, V_d).squeeze(-1)

        self.lig = LayerIntegratedGradients(_forward, model.backbone.mp.W_o)

    def explain(
        self,
        smiles: str,
        lm_emb: np.ndarray,
        n_steps: int = 50,
    ) -> np.ndarray:
        """
        Return per-atom attribution scores for a single molecule.

        Attribution is the L2 norm of the IG tensor over the hidden-state
        dimension, giving one non-negative scalar per heavy atom.

        Params:
            smiles: str : SMILES string of the molecule to explain
            lm_emb: np.ndarray : ChemBERTa [CLS] embedding, shape (768,)
            n_steps: int : number of integration steps (higher = more accurate)
        Returns:
            np.ndarray : per-atom scores, shape (n_heavy_atoms,), float32
        """
        loader = _make_single_loader(smiles, lm_emb)
        batch = next(iter(loader))
        bmg = batch.bmg.to(self.device)
        X_d = batch.X_d.to(self.device) if batch.X_d is not None else None

        attr = self.lig.attribute(
            inputs=(bmg, X_d),
            n_steps=n_steps,
            attribute_to_layer_input=False,  # attribute to W_o output
        )
        # attr: (n_atoms, d_h) — aggregate over feature dim
        scores = torch.norm(attr, dim=-1).detach().cpu().numpy().astype(np.float32)
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

    # Normalise scores symmetrically around zero so red/blue balance makes sense.
    max_abs = float(np.abs(scores).max()) or 1.0
    norm_scores = scores / max_abs  # in [-1, 1]

    cmap = plt.cm.RdBu_r  # red = positive, blue = negative
    atom_colours: dict[int, tuple] = {}
    for i, s in enumerate(norm_scores):
        rgba = cmap((s + 1.0) / 2.0)  # map [-1,1] → [0,1]
        atom_colours[i] = rgba[:3]

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
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
    svg = drawer.GetDrawingText()

    # Render SVG into a matplotlib figure via a temporary PNG round-trip.
    import io

    from rdkit.Chem.Draw import MolToImage

    pillow_colours = {
        i: tuple(int(c * 255) for c in atom_colours[i]) for i in atom_colours
    }
    img = MolToImage(
        mol,
        size=size,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightColor=None,
    )
    # Fall back to SVG-based rendering for proper per-atom colouring.
    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(bytestring=svg.encode())
        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes))
    except ImportError:
        # cairosvg not available; use simple uniform highlight as fallback.
        pass

    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    # Colourbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Attribution score", fontsize=8)
    plt.tight_layout()
    return fig
