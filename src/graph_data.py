from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from transformers import AutoModel, AutoTokenizer

_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBertaEncoder(nn.Module):
    """Frozen ChemBERTa encoder that extracts per-molecule [CLS] embeddings."""

    def __init__(self, model_name: str = _CHEMBERTA_MODEL) -> None:
        """
        Load and permanently freeze ChemBERTa weights.

        Params:
            model_name: str : HuggingFace model identifier
        Returns:
            None
        """
        import transformers

        super().__init__()
        # Suppress the MISSING/UNEXPECTED weight report — expected when loading the
        # base encoder from a masked-LM checkpoint without the LM head.
        transformers.logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        transformers.logging.set_verbosity_warning()
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

    @torch.no_grad()
    def encode(self, smiles: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode SMILES strings into [CLS] hidden-state vectors.

        Params:
            smiles: list[str] : SMILES strings to encode
            batch_size: int : molecules per forward pass
        Returns:
            np.ndarray : float32 array of shape (n, 768)
        """
        device = next(self.bert.parameters()).device
        chunks = []
        for i in range(0, len(smiles), batch_size):
            batch = smiles[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.bert(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            chunks.append(cls)
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def forward(self, smiles: list[str]) -> torch.Tensor:
        """
        Encode SMILES and return embeddings as a Tensor (nn.Module compatibility).

        Params:
            smiles: list[str] : SMILES strings to encode
        Returns:
            torch.Tensor : shape (n, 768)
        """
        return torch.from_numpy(self.encode(smiles))


def build_chemprop_dataset(
    smiles: pd.Series | list[str],
    targets: np.ndarray,
    lm_embeddings: np.ndarray,
) -> MoleculeDataset:
    """
    Build a chemprop MoleculeDataset with ChemBERTa embeddings as extra descriptors.

    Each datapoint stores the ChemBERTa [CLS] embedding in x_d, which the
    chemprop DataLoader batches into a V_d tensor alongside the molecular graphs.
    Embeddings are precomputed at construction time — safe because ChemBertaEncoder
    is frozen throughout training.

    Params:
        smiles: pd.Series | list[str] : SMILES strings
        targets: np.ndarray : regression targets, shape (n,)
        lm_embeddings: np.ndarray : ChemBERTa [CLS] embeddings, shape (n, 768)
    Returns:
        MoleculeDataset : dataset ready for chemprop.data.build_dataloader
    """
    smiles_list = list(smiles)
    datapoints = [
        MoleculeDatapoint.from_smi(
            smi,
            y=np.array([float(y)], dtype=np.float32),
            x_d=lm_emb.astype(np.float32),
        )
        for smi, y, lm_emb in zip(smiles_list, targets, lm_embeddings)
    ]
    return MoleculeDataset(datapoints)
