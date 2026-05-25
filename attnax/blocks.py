# SPDX-License-Identifier: Apache-2.0

"""Transformer encoder and decoder blocks."""

from __future__ import annotations

from typing import Optional, Union, overload

import flax.nnx as nnx
import jax.numpy as jnp

from .attention import MultiHeadAttention
from .cache import KVLayerCache
from .feedforward import FeedForward
from .config import AttentionType, FfActivation, NormKind
from .kernels import ScoreMod
from .norms import create_norm


class EncoderBlock(nnx.Module):
  """Transformer encoder block.

  Self-attention sub-layer followed by a feed-forward sub-layer, each
  with a residual connection and normalisation. When ``pre_norm`` is
  ``True`` the layout is

  .. code-block:: text

      x = x + self_attn(norm1(x))
      x = x + ffn(norm2(x))

  otherwise normalisation is applied after the residual sum.

  Args:
    rngs: Flax NNX random key container.
    d_model: Model dimensionality.
    num_heads: Number of query attention heads.
    d_ff: Feed-forward hidden width.
    dropout_rate: Dropout probability.
    pre_norm: Pre-norm layout when ``True``, post-norm otherwise.
    attention_type: Attention backend.
    attention_block_size: Block size for memory-efficient, flash and
      pallas_flash backends.
    linear_attention_chunk_size: Chunk size for the linear backend.
    norm_type: Normalisation kind.
    ff_activation: Feed-forward activation.
    num_kv_heads: KV heads for GQA/MQA.
    attention_window: Causal sliding-window size for self-attention.
    score_mod: :data:`~attnax.kernels.ScoreMod` applied to
      self-attention.
    use_rope: Apply RoPE in self-attention.
    rope_base: RoPE base :math:`\\theta`.
    rope_max_positions: Length of the RoPE table.
  """

  def __init__(
    self,
    rngs: nnx.Rngs,
    d_model: int,
    num_heads: int,
    d_ff: int,
    dropout_rate: float = 0.1,
    pre_norm: bool = True,
    attention_type: AttentionType = AttentionType.STANDARD,
    attention_block_size: int = 512,
    linear_attention_chunk_size: int = 64,
    norm_type: NormKind = "layer",
    ff_activation: FfActivation = "gelu",
    num_kv_heads: Optional[int] = None,
    attention_window: Optional[int] = None,
    score_mod: Optional[ScoreMod] = None,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    rope_max_positions: int = 8192,
  ):
    self.pre_norm = pre_norm

    self.self_attn = MultiHeadAttention(
      rngs=rngs,
      in_features=d_model,
      num_heads=num_heads,
      num_kv_heads=num_kv_heads,
      dropout_rate=dropout_rate,
      attention_type=attention_type,
      attention_block_size=attention_block_size,
      linear_attention_chunk_size=linear_attention_chunk_size,
      attention_window=attention_window,
      score_mod=score_mod,
      use_rope=use_rope,
      rope_base=rope_base,
      rope_max_positions=rope_max_positions,
    )
    self.ffn = FeedForward(
      rngs,
      d_model=d_model,
      d_ff=d_ff,
      dropout_rate=dropout_rate,
      ff_activation=ff_activation,
    )
    self.ln1 = create_norm(norm_type, d_model, rngs=rngs)
    self.ln2 = create_norm(norm_type, d_model, rngs=rngs)

  @overload
  def __call__(
    self,
    x: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    self_attn_kv_cache: None = None,
  ) -> jnp.ndarray: ...

  @overload
  def __call__(
    self,
    x: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    self_attn_kv_cache: KVLayerCache,
  ) -> tuple[jnp.ndarray, KVLayerCache]: ...

  def __call__(
    self,
    x: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids: Optional[jnp.ndarray] = None,
    self_attn_kv_cache: Optional[KVLayerCache] = None,
  ) -> Union[jnp.ndarray, tuple[jnp.ndarray, KVLayerCache]]:
    """Applies the encoder block.

    Args:
      x: Input of shape ``(batch, seq_len, d_model)``.
      mask: Boolean attention mask for self-attention.
      deterministic: If ``True``, disables dropout.
      position_ids: Integer positions of shape ``(batch, seq_len)``
        for RoPE.
      self_attn_kv_cache: :class:`KVLayerCache` for self-attention.

    Returns:
      Output of shape ``(batch, seq_len, d_model)``, or
      ``(output, updated_cache)`` when ``self_attn_kv_cache`` is set.
    """
    residual = x
    y = self.ln1(x) if self.pre_norm else x
    if self_attn_kv_cache is not None:
      y, kv_out = self.self_attn(
        y,
        mask=mask,
        deterministic=deterministic,
        position_ids_q=position_ids,
        position_ids_kv=position_ids,
        kv_cache=self_attn_kv_cache,
      )
    else:
      y = self.self_attn(
        y,
        mask=mask,
        deterministic=deterministic,
        position_ids_q=position_ids,
        position_ids_kv=position_ids,
      )
      kv_out = None
    x = residual + y
    if not self.pre_norm:
      x = self.ln1(x)

    residual = x
    y = self.ln2(x) if self.pre_norm else x
    y = self.ffn(y, deterministic=deterministic)
    x = residual + y
    if not self.pre_norm:
      x = self.ln2(x)

    if self_attn_kv_cache is not None:
      return x, kv_out
    return x


class DecoderBlock(nnx.Module):
  """Transformer decoder block.

  Three sub-layers — self-attention, cross-attention, feed-forward —
  with residual connections. KV caching is supported on
  self-attention only.

  Args:
    rngs: Flax NNX random key container.
    d_model: Model dimensionality.
    num_heads: Number of query attention heads.
    d_ff: Feed-forward hidden width.
    dropout_rate: Dropout probability.
    pre_norm: Pre-norm layout when ``True``, post-norm otherwise.
    attention_type: Attention backend.
    attention_block_size: Block size for memory-efficient, flash and
      pallas_flash backends.
    linear_attention_chunk_size: Chunk size for the linear backend.
    norm_type: Normalisation kind.
    ff_activation: Feed-forward activation.
    num_kv_heads: KV heads for GQA/MQA in self-attention.
    attention_window: Causal sliding-window size for self-attention.
    score_mod: :data:`~attnax.kernels.ScoreMod` applied to
      self-attention.
    use_rope: Apply RoPE in self-attention.
    rope_base: RoPE base :math:`\\theta`.
    rope_max_positions: Length of the RoPE table.
  """

  def __init__(
    self,
    rngs: nnx.Rngs,
    d_model: int,
    num_heads: int,
    d_ff: int,
    dropout_rate: float = 0.1,
    pre_norm: bool = True,
    attention_type: AttentionType = AttentionType.STANDARD,
    attention_block_size: int = 512,
    linear_attention_chunk_size: int = 64,
    norm_type: NormKind = "layer",
    ff_activation: FfActivation = "gelu",
    num_kv_heads: Optional[int] = None,
    attention_window: Optional[int] = None,
    score_mod: Optional[ScoreMod] = None,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    rope_max_positions: int = 8192,
  ):
    self.pre_norm = pre_norm

    self.self_attn = MultiHeadAttention(
      rngs=rngs,
      in_features=d_model,
      num_heads=num_heads,
      num_kv_heads=num_kv_heads,
      dropout_rate=dropout_rate,
      attention_type=attention_type,
      attention_block_size=attention_block_size,
      linear_attention_chunk_size=linear_attention_chunk_size,
      attention_window=attention_window,
      score_mod=score_mod,
      use_rope=use_rope,
      rope_base=rope_base,
      rope_max_positions=rope_max_positions,
    )
    self.cross_attn = MultiHeadAttention(
      rngs=rngs,
      in_features=d_model,
      num_heads=num_heads,
      num_kv_heads=num_kv_heads,
      dropout_rate=dropout_rate,
      attention_type=attention_type,
      attention_block_size=attention_block_size,
      linear_attention_chunk_size=linear_attention_chunk_size,
      use_rope=use_rope,
      rope_base=rope_base,
      rope_max_positions=rope_max_positions,
    )
    self.ffn = FeedForward(
      rngs,
      d_model=d_model,
      d_ff=d_ff,
      dropout_rate=dropout_rate,
      ff_activation=ff_activation,
    )
    self.ln1 = create_norm(norm_type, d_model, rngs=rngs)
    self.ln2 = create_norm(norm_type, d_model, rngs=rngs)
    self.ln3 = create_norm(norm_type, d_model, rngs=rngs)

  @overload
  def __call__(
    self,
    x: jnp.ndarray,
    *,
    encoder_output: jnp.ndarray,
    self_mask: Optional[jnp.ndarray] = None,
    cross_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids_self: Optional[jnp.ndarray] = None,
    position_ids_cross_q: Optional[jnp.ndarray] = None,
    position_ids_cross_kv: Optional[jnp.ndarray] = None,
    self_attn_kv_cache: None = None,
  ) -> jnp.ndarray: ...

  @overload
  def __call__(
    self,
    x: jnp.ndarray,
    *,
    encoder_output: jnp.ndarray,
    self_mask: Optional[jnp.ndarray] = None,
    cross_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids_self: Optional[jnp.ndarray] = None,
    position_ids_cross_q: Optional[jnp.ndarray] = None,
    position_ids_cross_kv: Optional[jnp.ndarray] = None,
    self_attn_kv_cache: KVLayerCache,
  ) -> tuple[jnp.ndarray, KVLayerCache]: ...

  def __call__(
    self,
    x: jnp.ndarray,
    *,
    encoder_output: jnp.ndarray,
    self_mask: Optional[jnp.ndarray] = None,
    cross_mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids_self: Optional[jnp.ndarray] = None,
    position_ids_cross_q: Optional[jnp.ndarray] = None,
    position_ids_cross_kv: Optional[jnp.ndarray] = None,
    self_attn_kv_cache: Optional[KVLayerCache] = None,
  ) -> Union[jnp.ndarray, tuple[jnp.ndarray, KVLayerCache]]:
    """Applies the decoder block.

    Args:
      x: Input of shape ``(batch, seq_q, d_model)``.
      encoder_output: Encoder output for cross-attention of shape
        ``(batch, seq_kv, d_model)``.
      self_mask: Boolean mask for self-attention.
      cross_mask: Boolean mask for cross-attention.
      deterministic: If ``True``, disables dropout.
      position_ids_self: RoPE positions for self-attention.
      position_ids_cross_q: RoPE positions on the cross-attention
        query side.
      position_ids_cross_kv: RoPE positions on the cross-attention
        key/value side.
      self_attn_kv_cache: :class:`KVLayerCache` for self-attention.

    Returns:
      Output of shape ``(batch, seq_q, d_model)``, or
      ``(output, updated_cache)`` when ``self_attn_kv_cache`` is set.
    """
    residual = x
    y = self.ln1(x) if self.pre_norm else x
    if self_attn_kv_cache is not None:
      y, kv_out = self.self_attn(
        y,
        mask=self_mask,
        deterministic=deterministic,
        position_ids_q=position_ids_self,
        position_ids_kv=position_ids_self,
        kv_cache=self_attn_kv_cache,
      )
    else:
      y = self.self_attn(
        y,
        mask=self_mask,
        deterministic=deterministic,
        position_ids_q=position_ids_self,
        position_ids_kv=position_ids_self,
      )
      kv_out = None
    x = residual + y
    if not self.pre_norm:
      x = self.ln1(x)

    residual = x
    y = self.ln2(x) if self.pre_norm else x
    y = self.cross_attn(
      y,
      context=encoder_output,
      mask=cross_mask,
      deterministic=deterministic,
      position_ids_q=position_ids_cross_q,
      position_ids_kv=position_ids_cross_kv,
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

    if self_attn_kv_cache is not None:
      return x, kv_out
    return x
