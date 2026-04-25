from __future__ import annotations

import torch
import torch.nn as nn
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing


class ChempropBackbone(nn.Module):
    """
    Wraps chemprop's BondMessagePassing to expose per-atom hidden states.

    By default initialises a fresh BondMessagePassing with chemprop's standard
    MultiHot atom (72-dim) and bond (14-dim) featurisers. If a checkpoint path
    is supplied the full MPNN is loaded and only the message-passing submodule
    is retained; the chemprop aggregation and FFN head are discarded because we
    replace them with AttentionPooling and FusionMLP.
    """

    # Feature dims that match chemprop v2's default MultiHot featurisers.
    _D_V = 72
    _D_E = 14

    def __init__(
        self,
        d_h: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        checkpoint_path: str | None = None,
    ) -> None:
        """
        Initialise the backbone.

        Params:
            d_h: int : hidden dimension (ignored when loading from checkpoint)
            depth: int : number of message-passing steps (ignored from checkpoint)
            dropout: float : dropout probability inside the MPNN
            checkpoint_path: str | None : path to a chemprop v2 .ckpt file
        Returns:
            None
        """
        super().__init__()
        if checkpoint_path is not None:
            full_model = MPNN.load_from_checkpoint(checkpoint_path)
            self.mp = full_model.message_passing
        else:
            self.mp = BondMessagePassing(
                d_v=self._D_V,
                d_e=self._D_E,
                d_h=d_h,
                depth=depth,
                dropout=dropout,
            )
        # Infer actual hidden dim from the final linear projection weight.
        # BondMessagePassing names this layer W_o; adjust if the chemprop
        # version in use differs.
        self.d_h: int = self.mp.W_o.out_features

    def forward(self, bmg) -> torch.Tensor:
        """
        Run directed message passing and return per-atom hidden states.

        Params:
            bmg: BatchMolGraph : batched molecular graph from chemprop DataLoader
        Returns:
            torch.Tensor : per-atom hidden states, shape (n_atoms_total, d_h)
        """
        return self.mp(bmg)


class AttentionPooling(nn.Module):
    """
    Sigmoid-gated attention pooling from atom hidden states to a molecule vector.

    Uses sigmoid rather than softmax so multiple atoms can simultaneously receive
    high weight; this avoids zero-sum competition between atoms and produces
    cleaner per-atom attribution signals when Captum IntegratedGradients is applied
    downstream.
    """

    def __init__(self, d_h: int) -> None:
        """
        Initialise the attention gate.

        Params:
            d_h: int : atom hidden-state dimension
        Returns:
            None
        """
        super().__init__()
        self.gate = nn.Linear(d_h, 1)

    def forward(self, H: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Pool atom representations to molecule representations.

        Params:
            H: torch.Tensor : atom hidden states, shape (n_atoms_total, d_h)
            batch: torch.Tensor : molecule index per atom, shape (n_atoms_total,), int64
        Returns:
            torch.Tensor : molecule embeddings, shape (n_mols, d_h)
        """
        scores = torch.sigmoid(self.gate(H))  # (n_atoms, 1)
        weighted = scores * H  # (n_atoms, d_h)
        n_mols = int(batch.max().item()) + 1
        out = torch.zeros(n_mols, H.size(-1), device=H.device, dtype=H.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted), weighted)
        return out


class FusionMLP(nn.Module):
    """MLP that fuses the graph embedding with the ChemBERTa [CLS] embedding."""

    def __init__(self, d_in: int, d_hidden: int = 256, dropout: float = 0.1) -> None:
        """
        Initialise the fusion head.

        Params:
            d_in: int : concatenated input dimension (d_graph + d_lm)
            d_hidden: int : intermediate hidden dimension
            dropout: float : dropout probability
        Returns:
            None
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict logD from fused molecular embedding.

        Params:
            x: torch.Tensor : fused embedding, shape (n_mols, d_in)
        Returns:
            torch.Tensor : predictions, shape (n_mols, 1)
        """
        return self.net(x)


class LipophilicityGNN(nn.Module):
    """
    Full model: chemprop MPNN backbone + attention pooling + ChemBERTa fusion.

    Architecture (Option A):
      1. ChempropBackbone  : SMILES graph -> per-atom hidden states  h_atoms
      2. AttentionPooling  : h_atoms -> graph embedding              h_graph
      3. Concatenation     : [h_graph ; V_d]                        h_fused
      4. FusionMLP         : h_fused -> predicted logD scalar
    """

    def __init__(
        self,
        d_h: int = 300,
        d_lm: int = 768,
        d_hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
        checkpoint_path: str | None = None,
    ) -> None:
        """
        Initialise all submodules.

        Params:
            d_h: int : MPNN hidden dimension
            d_lm: int : ChemBERTa embedding dimension
            d_hidden: int : FusionMLP hidden dimension
            depth: int : number of message-passing steps
            dropout: float : dropout in MPNN and FusionMLP
            checkpoint_path: str | None : optional chemprop checkpoint to load
        Returns:
            None
        """
        super().__init__()
        self.backbone = ChempropBackbone(
            d_h=d_h, depth=depth, dropout=dropout, checkpoint_path=checkpoint_path
        )
        self.pool = AttentionPooling(self.backbone.d_h)
        self.fusion = FusionMLP(
            d_in=self.backbone.d_h + d_lm,
            d_hidden=d_hidden,
            dropout=dropout,
        )

    def forward(self, bmg, V_d: torch.Tensor) -> torch.Tensor:
        """
        Predict logD for a batch of molecules.

        Params:
            bmg: BatchMolGraph : batched molecular graph from chemprop DataLoader
            V_d: torch.Tensor : ChemBERTa [CLS] embeddings, shape (n_mols, d_lm)
        Returns:
            torch.Tensor : predicted logD values, shape (n_mols, 1)
        """
        H = self.backbone(bmg)  # (n_atoms_total, d_h)
        h_graph = self.pool(H, bmg.batch)  # (n_mols, d_h)
        h_fused = torch.cat([h_graph, V_d], dim=-1)  # (n_mols, d_h + d_lm)
        return self.fusion(h_fused)  # (n_mols, 1)
