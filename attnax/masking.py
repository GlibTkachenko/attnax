# SPDX-License-Identifier: Apache-2.0

"""Attention masking utilities."""

from __future__ import annotations

import jax.numpy as jnp


def make_padding_mask(input_ids: jnp.ndarray, pad_token_id: int = 0) -> jnp.ndarray:
    """Creates padding mask from input token IDs.

    Args:
      input_ids: Token IDs of shape (batch, seq_len).
      pad_token_id: Token ID representing padding.

    Returns:
      Boolean mask of shape (batch, 1, 1, seq_len) where True indicates
      positions to attend to.
    """
    mask = input_ids != pad_token_id
    return mask[:, None, None, :]


def make_causal_mask(seq_len: int) -> jnp.ndarray:
    """Creates causal mask for autoregressive attention.

    Args:
      seq_len: Sequence length.

    Returns:
      Boolean mask of shape (1, 1, seq_len, seq_len) where True indicates
      positions that can be attended to (lower triangular).
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return mask[None, None, :, :]


def combine_masks(*masks: jnp.ndarray | None) -> jnp.ndarray | None:
    """Combines multiple boolean masks via logical AND.

    Args:
      *masks: Variable number of boolean masks or None.

    Returns:
      Combined mask or None if all inputs are None.
    """
    result = None
    for mask in masks:
        if mask is None:
            continue
        result = mask if result is None else (result & mask)
    return result
