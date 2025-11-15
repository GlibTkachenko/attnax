# SPDX-License-Identifier: Apache-2.0

"""Token and positional embeddings."""

from __future__ import annotations

import jax.numpy as jnp
import flax.nnx as nnx


class TokenEmbedding(nnx.Module):
    """Token embedding layer."""

    def __init__(self, rngs: nnx.Rngs, vocab_size: int, d_model: int):
        self.embed = nnx.Embed(num_embeddings=vocab_size, features=d_model, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """Embeds token IDs.

        Args:
          token_ids: Token IDs of shape (batch, seq_len).

        Returns:
          Embeddings of shape (batch, seq_len, d_model).
        """
        return self.embed(token_ids)


class PositionalEncoding(nnx.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, max_len: int, d_model: int):
        self.max_len = max_len
        self.d_model = d_model
        self.positional = self._create_sinusoidal_positions(max_len, d_model)

    @staticmethod
    def _create_sinusoidal_positions(max_len: int, d_model: int) -> jnp.ndarray:
        """Creates sinusoidal positional encodings.

        Args:
          max_len: Maximum sequence length.
          d_model: Model dimension.

        Returns:
          Positional encodings of shape (max_len, d_model).
        """
        positions = jnp.arange(max_len)[:, None]
        dims = jnp.arange(d_model)[None, :]
        angle_rates = 1.0 / (10000 ** (2 * (dims // 2) / d_model))
        angles = positions * angle_rates
        pos_encoding = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))
        return pos_encoding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Adds positional encoding to embeddings.

        Args:
          x: Input embeddings of shape (batch, seq_len, d_model).

        Returns:
          Embeddings with added positional encoding.
        """
        seq_len = x.shape[1]
        return x + self.positional[None, :seq_len, :]
