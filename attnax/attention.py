# SPDX-License-Identifier: Apache-2.0

"""Multi-head attention."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import flax.nnx as nnx


class MultiHeadAttentionLayer(nnx.Module):
    """Multi-head attention with support for self and cross attention."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        num_heads: int,
        in_features: int,
        qkv_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout_rate: float = 0.0,
        broadcast_dropout: bool = True,
        decode: bool = False,
    ):
        if out_features is None:
            out_features = in_features
        if qkv_features is None:
            qkv_features = in_features

        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            qkv_features=qkv_features,
            out_features=out_features,
            dropout_rate=dropout_rate,
            broadcast_dropout=broadcast_dropout,
            decode=decode,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        context: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
    ) -> jnp.ndarray:
        """Applies multi-head attention.

        Args:
          x: Query input of shape (batch, seq_q, in_features).
          context: Optional context for cross-attention of shape
            (batch, seq_kv, in_features). If None, performs self-attention.
          mask: Optional boolean mask broadcastable to
            (batch, num_heads, seq_q, seq_kv).
          deterministic: If True, disables dropout. If None, uses module default.

        Returns:
          Output of shape (batch, seq_q, out_features).
        """
        inputs_k = context if context is not None else None
        inputs_v = context if context is not None else None

        return self.mha(
            inputs_q=x,
            inputs_k=inputs_k,
            inputs_v=inputs_v,
            mask=mask,
            deterministic=deterministic,
        )
