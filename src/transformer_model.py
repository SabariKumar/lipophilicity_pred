from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
_TOKENIZER_CACHE: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer:
    """
    Return the ChemBERTa tokenizer, loading it once and caching it in-process.

    Params:
        None
    Returns:
        AutoTokenizer : ChemBERTa tokenizer
    """
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    return _TOKENIZER_CACHE


def tokenize(smiles: list[str], max_length: int = 128) -> dict[str, torch.Tensor]:
    """
    Tokenize a list of SMILES strings for ChemBERTa.

    Params:
        smiles: list[str] : SMILES strings to tokenize
        max_length: int : maximum token sequence length (default 128)
    Returns:
        dict[str, torch.Tensor] : 'input_ids' and 'attention_mask' tensors,
            shape (n, max_length)
    """
    tokenizer = get_tokenizer()
    return tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# Regression heads
# ---------------------------------------------------------------------------


class QM9PretrainHead(nn.Module):
    """Single linear layer mapping [CLS] → n_targets normalised QM9 values."""

    def __init__(self, d_model: int, n_targets: int) -> None:
        """
        Initialise the QM9 pretraining head.

        Params:
            d_model: int : ChemBERTa hidden dimension (768)
            n_targets: int : number of QM9 regression targets
        Returns:
            None
        """
        super().__init__()
        self.linear = nn.Linear(d_model, n_targets)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Project [CLS] to QM9 target predictions.

        Params:
            cls_token: torch.Tensor : shape (batch, d_model)
        Returns:
            torch.Tensor : shape (batch, n_targets)
        """
        return self.linear(cls_token)


class LogDFinetuneHead(nn.Module):
    """MLP head mapping [CLS] → scalar logD prediction."""

    def __init__(self, d_model: int, d_hidden: int = 256, dropout: float = 0.1) -> None:
        """
        Initialise the logD fine-tuning head.

        Params:
            d_model: int : ChemBERTa hidden dimension (768)
            d_hidden: int : MLP hidden layer size
            dropout: float : dropout probability
        Returns:
            None
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Project [CLS] to a scalar logD prediction.

        Params:
            cls_token: torch.Tensor : shape (batch, d_model)
        Returns:
            torch.Tensor : shape (batch, 1)
        """
        return self.net(cls_token)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SMILESTransformer(nn.Module):
    """
    ChemBERTa backbone with a swappable regression head.

    The backbone is always unfrozen — for pretraining (QM9 multi-task) the
    head is a QM9PretrainHead; after phase 1 the head is swapped for a
    LogDFinetuneHead and the whole model is fine-tuned end-to-end on logD.

    The model accepts pre-tokenized inputs (input_ids, attention_mask) to
    keep tokenization out of the forward pass and avoid re-loading the
    tokenizer on each batch.
    """

    def __init__(self, head: nn.Module, model_name: str = CHEMBERTA_MODEL) -> None:
        """
        Initialise the transformer with ChemBERTa backbone and a head.

        Params:
            head: nn.Module : QM9PretrainHead or LogDFinetuneHead
            model_name: str : HuggingFace model ID for the backbone
        Returns:
            None
        """
        super().__init__()
        # Load as ForMaskedLM so weight-tied embeddings are correctly resolved
        # (safetensors de-duplicates embeddings.word_embeddings.weight ↔
        # lm_head.decoder.weight; AutoModel would leave the embedding table
        # randomly initialized).  We extract only the encoder backbone and
        # discard the LM head.
        import transformers as _hf

        _prev_level = _hf.logging.get_verbosity()
        _hf.logging.set_verbosity_error()
        _full = AutoModelForMaskedLM.from_pretrained(model_name)
        _hf.logging.set_verbosity(_prev_level)
        self.backbone = _full.roberta
        del _full
        self.head = head
        self.train()

    @property
    def d_model(self) -> int:
        """
        Return the backbone hidden dimension.

        Params:
            None
        Returns:
            int : hidden size of the ChemBERTa backbone
        """
        return self.backbone.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run backbone and head, returning predictions.

        Params:
            input_ids: torch.Tensor : shape (batch, seq_len)
            attention_mask: torch.Tensor : shape (batch, seq_len)
        Returns:
            torch.Tensor : head output, shape depends on head type
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.head(cls)

    def swap_head(self, new_head: nn.Module) -> None:
        """
        Replace the current head in-place.

        Params:
            new_head: nn.Module : replacement head module
        Returns:
            None
        """
        self.head = new_head
