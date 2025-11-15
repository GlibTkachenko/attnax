# SPDX-License-Identifier: Apache-2.0

"""Transformer encoder and decoder blocks."""

from __future__ import annotations

from typing import Optional

import flax.nnx as nnx
import jax.numpy as jnp

from .attention import MultiHeadAttentionLayer
from .feedforward import FeedForward


class EncoderBlock(nnx.Module):
    """Single transformer encoder block with self-attention and FFN."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        pre_norm: bool = True,
        activation: str = "relu",
    ):
        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttentionLayer(
            rngs=rngs,
            in_features=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.ffn = FeedForward(
            rngs,
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
    ) -> jnp.ndarray:
        """Applies encoder block transformation.

        Args:
          x: Input of shape (batch, seq_len, d_model).
          mask: Optional attention mask.
          deterministic: If True, disables dropout.

        Returns:
          Output of shape (batch, seq_len, d_model).
        """
        residual = x
        y = self.ln1(x) if self.pre_norm else x
        y = self.self_attn(y, mask=mask, deterministic=deterministic)
        x = residual + y
        if not self.pre_norm:
            x = self.ln1(x)

        residual = x
        y = self.ln2(x) if self.pre_norm else x
        y = self.ffn(y, deterministic=deterministic)
        x = residual + y
        if not self.pre_norm:
            x = self.ln2(x)

        return x


class DecoderBlock(nnx.Module):
    """Single transformer decoder block with self-attention,
    cross-attention, and FFN."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        pre_norm: bool = True,
        activation: str = "relu",
    ):
        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttentionLayer(
            rngs=rngs,
            in_features=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.cross_attn = MultiHeadAttentionLayer(
            rngs=rngs,
            in_features=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.ffn = FeedForward(
            rngs,
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln3 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        encoder_output: jnp.ndarray,
        self_mask: Optional[jnp.ndarray] = None,
        cross_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
    ) -> jnp.ndarray:
        """Applies decoder block transformation.

        Args:
          x: Input of shape (batch, seq_len, d_model).
          encoder_output: Encoder output for cross-attention.
          self_mask: Optional self-attention mask (typically causal).
          cross_mask: Optional cross-attention mask.
          deterministic: If True, disables dropout.

        Returns:
          Output of shape (batch, seq_len, d_model).
        """
        residual = x
        y = self.ln1(x) if self.pre_norm else x
        y = self.self_attn(y, mask=self_mask, deterministic=deterministic)
        x = residual + y
        if not self.pre_norm:
            x = self.ln1(x)

        residual = x
        y = self.ln2(x) if self.pre_norm else x
        y = self.cross_attn(
            y, context=encoder_output, mask=cross_mask, deterministic=deterministic
        )
        x = residual + y
        if not self.pre_norm:
            x = self.ln2(x)

        residual = x
        y = self.ln3(x) if self.pre_norm else x
        y = self.ffn(y, deterministic=deterministic)
        x = residual + y
        if not self.pre_norm:
            x = self.ln3(x)

        return x
