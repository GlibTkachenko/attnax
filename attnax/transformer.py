# SPDX-License-Identifier: Apache-2.0

"""Transformer encoder and decoder wrappers."""

from __future__ import annotations

from typing import Optional, Union, overload

import flax.nnx as nnx
import jax.numpy as jnp

from .blocks import EncoderBlock
from .cache import KVLayerCache
from .config import TransformerConfig
from .embeddings import PositionalEncoding, TokenEmbedding
from .kernels.score_mods import alibi_mod
from .masking import combine_masks
from .norms import create_norm


def _build_layers(
  rngs: nnx.Rngs, config: TransformerConfig
) -> nnx.List:
  """Builds the stack of :class:`EncoderBlock` layers."""
  use_rope = config.pos_emb_type == "rope"
  rope_table = config.rope_max_positions or config.max_len
  score_mod = (
    alibi_mod(num_heads=config.num_heads)
    if config.pos_emb_type == "alibi"
    else None
  )
  return nnx.List(
    [
      EncoderBlock(
        rngs,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout_rate=config.dropout_rate,
        pre_norm=config.use_pre_norm,
        attention_type=config.attention_type,
        attention_block_size=config.attention_block_size,
        linear_attention_chunk_size=config.linear_attention_chunk_size,
        norm_type=config.norm_type,
        ff_activation=config.ff_activation,
        num_kv_heads=config.num_kv_heads,
        attention_window=config.attention_window,
        score_mod=score_mod,
        use_rope=use_rope,
        rope_base=config.rope_base,
        rope_max_positions=rope_table,
      )
      for _ in range(config.num_layers)
    ]
  )


class TransformerEncoder(nnx.Module):
  """Transformer encoder stack.

  Token embedding, positional encoding, ``config.num_layers`` encoder
  blocks, and a final normalisation. No mask is applied internally;
  pass ``padding_mask`` as needed.

  Args:
    rngs: Flax NNX random key container.
    config: Transformer hyperparameters.
  """

  def __init__(self, rngs: nnx.Rngs, config: TransformerConfig):
    self.config = config

    self.token_embed = TokenEmbedding(rngs, config.vocab_size, config.d_model)
    self.pos_encoding = PositionalEncoding(config.max_len, config.d_model)
    self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    self.layers = _build_layers(rngs, config)

    self.final_ln = create_norm(config.norm_type, config.d_model, rngs=rngs)

  @overload
  def __call__(
    self,
    input_ids: jnp.ndarray,
    *,
    padding_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    layer_kv_caches: None = None,
  ) -> jnp.ndarray: ...

  @overload
  def __call__(
    self,
    input_ids: jnp.ndarray,
    *,
    padding_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    layer_kv_caches: tuple[KVLayerCache, ...],
  ) -> tuple[jnp.ndarray, tuple[KVLayerCache, ...]]: ...

  def __call__(
    self,
    input_ids: jnp.ndarray,
    *,
    padding_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    layer_kv_caches: Optional[tuple[KVLayerCache, ...]] = None,
  ) -> Union[jnp.ndarray, tuple[jnp.ndarray, tuple[KVLayerCache, ...]]]:
    """Applies the encoder.

    Args:
      input_ids: Token ids of shape ``(batch, seq_len)``.
      padding_mask: Boolean key-padding mask.
      deterministic: If ``True``, disables dropout.
      position_ids: Integer positions of shape ``(batch, seq_len)``
        for RoPE.
      layer_kv_caches: Per-layer :class:`KVLayerCache` tuple of length
        ``config.num_layers``.

    Returns:
      Array of shape ``(batch, seq_len, d_model)``, or
      ``(output, updated_caches)`` when ``layer_kv_caches`` is set.
    """
    x = self.token_embed(input_ids)
    if self.config.pos_emb_type == "sinusoidal":
      x = self.pos_encoding(x)
    x = self.dropout(x, deterministic=deterministic)

    if layer_kv_caches is not None:
      if len(layer_kv_caches) != len(self.layers):
        raise ValueError(
          f"layer_kv_caches length {len(layer_kv_caches)} must match "
          f"num_layers {len(self.layers)}"
        )
      new_caches: list[KVLayerCache] = []
      for layer, kv in zip(self.layers, layer_kv_caches):
        out = layer(
          x,
          mask=padding_mask,
          deterministic=deterministic,
          position_ids=position_ids,
          self_attn_kv_cache=kv,
        )
        x, kv_next = out
        new_caches.append(kv_next)
      x = self.final_ln(x)
      return x, tuple(new_caches)

    for layer in self.layers:
      x = layer(
        x,
        mask=padding_mask,
        deterministic=deterministic,
        position_ids=position_ids,
      )

    return self.final_ln(x)


class TransformerDecoder(nnx.Module):
  """Decoder-only transformer stack.

  Token embedding, positional encoding, ``config.num_layers`` blocks,
  and a final normalisation. A causal mask is applied internally and
  AND-combined with ``padding_mask``. When ``layer_kv_caches`` is
  set, ``position_ids`` default to
  ``arange(past_len, past_len + seq_len)``.

  Args:
    rngs: Flax NNX random key container.
    config: Transformer hyperparameters.
  """

  def __init__(self, rngs: nnx.Rngs, config: TransformerConfig):
    self.config = config

    self.token_embed = TokenEmbedding(rngs, config.vocab_size, config.d_model)
    self.pos_encoding = PositionalEncoding(config.max_len, config.d_model)
    self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    self.layers = _build_layers(rngs, config)

    self.final_ln = create_norm(config.norm_type, config.d_model, rngs=rngs)

  @overload
  def __call__(
    self,
    input_ids: jnp.ndarray,
    *,
    padding_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    layer_kv_caches: None = None,
  ) -> jnp.ndarray: ...

  @overload
  def __call__(
    self,
    input_ids: jnp.ndarray,
    *,
    padding_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    layer_kv_caches: tuple[KVLayerCache, ...],
  ) -> tuple[jnp.ndarray, tuple[KVLayerCache, ...]]: ...

  def __call__(
    self,
    input_ids: jnp.ndarray,
    *,
    padding_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    layer_kv_caches: Optional[tuple[KVLayerCache, ...]] = None,
  ) -> Union[jnp.ndarray, tuple[jnp.ndarray, tuple[KVLayerCache, ...]]]:
    """Applies the decoder.

    Args:
      input_ids: Token ids of shape ``(batch, seq_len)``.
      padding_mask: Boolean key-padding mask of shape
        ``(batch, 1, 1, past_len + seq_len)``.
      deterministic: If ``True``, disables dropout.
      position_ids: Integer positions of shape ``(batch, seq_len)``
        for RoPE.
      layer_kv_caches: Per-layer :class:`KVLayerCache` tuple of length
        ``config.num_layers``.

    Returns:
      Array of shape ``(batch, seq_len, d_model)``, or
      ``(output, updated_caches)`` when ``layer_kv_caches`` is set.
    """
    batch, seq_q = input_ids.shape
    past_len = (
      int(layer_kv_caches[0].length) if layer_kv_caches is not None else 0
    )
    seq_kv = past_len + seq_q

    if position_ids is None:
      position_ids = jnp.broadcast_to(
        (jnp.arange(seq_q) + past_len)[None, :].astype(jnp.int32),
        (batch, seq_q),
      )

    q_idx = jnp.arange(seq_q) + past_len
    k_idx = jnp.arange(seq_kv)
    causal_mask = (q_idx[:, None] >= k_idx[None, :])[None, None, :, :]
    mask = combine_masks(padding_mask, causal_mask)

    x = self.token_embed(input_ids)
    if self.config.pos_emb_type == "sinusoidal":
      x = self.pos_encoding(x, start=past_len)
    x = self.dropout(x, deterministic=deterministic)

    if layer_kv_caches is not None:
      if len(layer_kv_caches) != len(self.layers):
        raise ValueError(
          f"layer_kv_caches length {len(layer_kv_caches)} must match "
          f"num_layers {len(self.layers)}"
        )
      new_caches: list[KVLayerCache] = []
      for layer, kv in zip(self.layers, layer_kv_caches):
        out = layer(
          x,
          mask=mask,
          deterministic=deterministic,
          position_ids=position_ids,
          self_attn_kv_cache=kv,
        )
        x, kv_next = out
        new_caches.append(kv_next)
      x = self.final_ln(x)
      return x, tuple(new_caches)

    for layer in self.layers:
      x = layer(
        x,
        mask=mask,
        deterministic=deterministic,
        position_ids=position_ids,
      )

    return self.final_ln(x)
