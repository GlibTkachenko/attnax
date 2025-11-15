# SPDX-License-Identifier: Apache-2.0

"""Attnax: Composable transformer components for JAX."""

__version__ = "0.1.0"

from .attention import MultiHeadAttentionLayer
from .blocks import DecoderBlock, EncoderBlock
from .config import TransformerConfig
from .embeddings import PositionalEncoding, TokenEmbedding
from .encoder import TransformerEncoder
from .feedforward import FeedForward
from .masking import combine_masks, make_causal_mask, make_padding_mask

__all__ = [
    "__version__",
    "MultiHeadAttentionLayer",
    "EncoderBlock",
    "DecoderBlock",
    "TransformerEncoder",
    "FeedForward",
    "TokenEmbedding",
    "PositionalEncoding",
    "TransformerConfig",
    "make_padding_mask",
    "make_causal_mask",
    "combine_masks",
]