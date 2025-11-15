# SPDX-License-Identifier: Apache-2.0

"""Position-wise feed-forward network."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx


class FeedForward(nnx.Module):
    """Two-layer MLP with configurable activation."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        self.dense1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.dense2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.activation = activation

    def __call__(
        self, x: jnp.ndarray, *, deterministic: Optional[bool] = None
    ) -> jnp.ndarray:
        """Applies position-wise feed-forward transformation.

        Args:
          x: Input of shape (batch, seq_len, d_model).
          deterministic: If True, disables dropout. If None, uses module default.

        Returns:
          Output of shape (batch, seq_len, d_model).
        """
        x = self.dense1(x)
        x = jax.nn.gelu(x) if self.activation == "gelu" else jax.nn.relu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x
