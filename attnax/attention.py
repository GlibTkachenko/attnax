# SPDX-License-Identifier: Apache-2.0

"""Multi-head attention with pluggable kernels and score-mod hooks."""

from __future__ import annotations

from typing import Optional, Union, overload
import functools

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .cache import KVLayerCache, update_kv_layer_cache
from .config import AttentionType
from .embeddings import apply_rope, rope_cos_sin_from_positions
from .kernels import (
  AttentionFn,
  ScoreMod,
  compose_score_mods,
  flash_attention,
  linear_attention,
  lite_attention,
  memory_efficient_attention,
  pallas_flash_attention,
  standard_attention,
)
from .kernels.score_mods import sliding_window_mod
from .masking import make_sliding_window_mask


def _repeat_kv_heads(
  x: jnp.ndarray, *, num_kv_heads: int, num_heads: int
) -> jnp.ndarray:
  """Repeats KV heads along axis 1 to match the query head count."""
  if num_kv_heads == num_heads:
    return x
  rep = num_heads // num_kv_heads
  return jnp.repeat(x, rep, axis=1)


def _select_builtin_kernel(
  attention_type: AttentionType,
  *,
  block_size: int,
  linear_chunk_size: int,
  gate_proj: Optional[nnx.Linear],
) -> AttentionFn:
  """Resolves an :class:`AttentionType` to a kernel callable."""
  if attention_type == AttentionType.STANDARD:
    return standard_attention
  if attention_type == AttentionType.MEMORY_EFFICIENT:
    return functools.partial(
      memory_efficient_attention, block_size=block_size
    )
  if attention_type == AttentionType.FLASH:
    return functools.partial(flash_attention, block_size=block_size)
  if attention_type == AttentionType.PALLAS_FLASH:
    return functools.partial(
      pallas_flash_attention,
      block_q=block_size,
      block_kv=block_size,
    )
  if attention_type == AttentionType.LINEAR:
    return functools.partial(
      linear_attention, chunk_size=linear_chunk_size
    )
  if attention_type == AttentionType.LITE:
    assert gate_proj is not None, (
      "AttentionType.LITE requires a gate projection"
    )

    def _lite_wrapped(
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray,
      *,
      mask: Optional[jnp.ndarray] = None,
      score_mod: Optional[ScoreMod] = None,
      dropout_rng: Optional[jax.Array] = None,
      dropout_rate: float = 0.0,
      deterministic: bool = True,
    ) -> jnp.ndarray:
      if score_mod is not None:
        raise NotImplementedError(
          "AttentionType.LITE does not support score_mod; switch to "
          "AttentionType.STANDARD or .MEMORY_EFFICIENT, or pass a custom "
          "attention_fn= that handles the gating itself."
        )
      return lite_attention(
        query,
        key,
        value,
        gate_proj,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
      )

    return _lite_wrapped
  raise ValueError(f"Unknown attention type: {attention_type}")


class MultiHeadAttention(nnx.Module):
  """Multi-head attention layer.

  Supports MHA, GQA (``1 < num_kv_heads < num_heads``) and MQA
  (``num_kv_heads == 1``); optional rotary position embeddings on Q
  and K; an attention backend selected by :class:`AttentionType` or a
  user-supplied :data:`~attnax.kernels.AttentionFn`; an optional
  :data:`~attnax.kernels.ScoreMod` and causal sliding-window; and an
  optional :class:`KVLayerCache` for autoregressive decoding.

  Args:
    rngs: Flax NNX random key container.
    num_heads: Number of query heads.
    in_features: Input dimensionality.
    qkv_features: QKV projection width. Defaults to ``in_features``.
    out_features: Output projection width. Defaults to ``in_features``.
    num_kv_heads: Number of key/value heads. Must divide ``num_heads``.
      Defaults to ``num_heads`` (MHA).
    dropout_rate: Output dropout probability.
    broadcast_dropout: Share the dropout mask across the batch.
    decode: Reserved; kept for API compatibility.
    attention_type: Built-in backend selection. Ignored when
      ``attention_fn`` is set.
    attention_block_size: Block size for ``memory_efficient``,
      ``flash`` and ``pallas_flash`` backends.
    linear_attention_chunk_size: Chunk size for the ``linear``
      backend.
    attention_fn: Custom kernel conforming to
      :data:`~attnax.kernels.AttentionFn`. Takes priority over
      ``attention_type``.
    score_mod: :data:`~attnax.kernels.ScoreMod` applied on every call.
    attention_window: Causal sliding-window size. When set, each query
      attends only to the most recent ``attention_window`` keys.
    use_rope: Apply rotary position embeddings to Q and K. Requires
      even ``head_dim``.
    rope_base: RoPE base :math:`\\theta`.
    rope_max_positions: Length of the precomputed RoPE table.

  Raises:
    ValueError: If ``qkv_features`` is not divisible by ``num_heads``,
      if ``num_kv_heads`` does not satisfy
      ``1 <= num_kv_heads <= num_heads`` and divide ``num_heads``, if
      ``use_rope`` is set with an odd ``head_dim``, or if both
      ``attention_fn`` and :attr:`AttentionType.LITE` are supplied.
  """

  def __init__(
    self,
    rngs: nnx.Rngs,
    *,
    num_heads: int,
    in_features: int,
    qkv_features: Optional[int] = None,
    out_features: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = True,
    decode: bool = False,
    attention_type: AttentionType = AttentionType.STANDARD,
    attention_block_size: int = 512,
    linear_attention_chunk_size: int = 64,
    attention_fn: Optional[AttentionFn] = None,
    score_mod: Optional[ScoreMod] = None,
    attention_window: Optional[int] = None,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    rope_max_positions: int = 8192,
  ):
    if out_features is None:
      out_features = in_features
    if qkv_features is None:
      qkv_features = in_features

    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    self.in_features = in_features
    self.qkv_features = qkv_features
    self.out_features = out_features
    self.dropout_rate = dropout_rate
    self.broadcast_dropout = broadcast_dropout
    self.decode = decode
    self.attention_type = attention_type
    self.attention_block_size = attention_block_size
    self.linear_attention_chunk_size = linear_attention_chunk_size
    self.use_rope = use_rope
    self.rope_base = rope_base
    self.rope_max_positions = rope_max_positions
    self.attention_window = attention_window

    head_dim = qkv_features // num_heads
    if qkv_features % num_heads != 0:
      raise ValueError(
        f"qkv_features ({qkv_features}) must be divisible by "
        f"num_heads ({num_heads})"
      )
    if self.num_kv_heads < 1 or self.num_kv_heads > num_heads:
      raise ValueError(
        f"num_kv_heads ({self.num_kv_heads}) must satisfy "
        f"1 <= num_kv_heads <= num_heads ({num_heads})"
      )
    if num_heads % self.num_kv_heads != 0:
      raise ValueError(
        f"num_heads ({num_heads}) must be divisible by "
        f"num_kv_heads ({self.num_kv_heads})"
      )
    if use_rope and head_dim % 2 != 0:
      raise ValueError(
        f"use_rope requires even head_dim, got head_dim={head_dim}"
      )
    if attention_fn is not None and attention_type == AttentionType.LITE:
      raise ValueError(
        "AttentionType.LITE owns a trainable gate projection and cannot "
        "be combined with a custom attention_fn. Either drop attention_fn "
        "or switch attention_type."
      )

    kv_features = head_dim * self.num_kv_heads

    self.query_proj = nnx.Linear(in_features, qkv_features, rngs=rngs)
    self.key_proj = nnx.Linear(in_features, kv_features, rngs=rngs)
    self.value_proj = nnx.Linear(in_features, kv_features, rngs=rngs)
    self.output_proj = nnx.Linear(qkv_features, out_features, rngs=rngs)

    self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    if attention_type == AttentionType.LITE and attention_fn is None:
      self.gate_proj = nnx.Linear(head_dim, 1, rngs=rngs)
    else:
      self.gate_proj = None

    if attention_fn is not None:
      self._attention_fn: AttentionFn = attention_fn
    else:
      self._attention_fn = _select_builtin_kernel(
        attention_type,
        block_size=attention_block_size,
        linear_chunk_size=linear_attention_chunk_size,
        gate_proj=self.gate_proj,
      )

    base_mods: list[ScoreMod] = []
    if attention_window is not None:
      base_mods.append(
        sliding_window_mod(window_size=attention_window, causal=True)
      )
    if score_mod is not None:
      base_mods.append(score_mod)
    self._base_score_mod: Optional[ScoreMod] = (
      compose_score_mods(*base_mods) if base_mods else None
    )

  @overload
  def __call__(
    self,
    x: jnp.ndarray,
    *,
    context: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids_q: Optional[jnp.ndarray] = None,
    position_ids_kv: Optional[jnp.ndarray] = None,
    score_mod: Optional[ScoreMod] = None,
    kv_cache: None = None,
  ) -> jnp.ndarray: ...

  @overload
  def __call__(
    self,
    x: jnp.ndarray,
    *,
    context: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids_q: Optional[jnp.ndarray] = None,
    position_ids_kv: Optional[jnp.ndarray] = None,
    score_mod: Optional[ScoreMod] = None,
    kv_cache: KVLayerCache,
  ) -> tuple[jnp.ndarray, KVLayerCache]: ...

  def __call__(
    self,
    x: jnp.ndarray,
    *,
    context: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
    position_ids_q: Optional[jnp.ndarray] = None,
    position_ids_kv: Optional[jnp.ndarray] = None,
    score_mod: Optional[ScoreMod] = None,
    kv_cache: Optional[KVLayerCache] = None,
  ) -> Union[jnp.ndarray, tuple[jnp.ndarray, KVLayerCache]]:
    """Applies attention to ``x``.

    Args:
      x: Query source of shape ``(batch, seq_q, in_features)``.
      context: Key/value source of shape
        ``(batch, seq_kv, in_features)`` for cross-attention. If
        ``None``, performs self-attention.
      mask: Boolean mask broadcastable to
        ``(batch, num_heads, seq_q, seq_kv)``.
      deterministic: If ``True``, disables dropout. If ``None``, uses
        the module's training/eval state.
      position_ids_q: Integer query positions of shape
        ``(batch, seq_q)`` for RoPE.
      position_ids_kv: Integer key/value positions for RoPE.
      score_mod: Per-call :data:`~attnax.kernels.ScoreMod` composed
        on top of any mod set at construction time.
      kv_cache: :class:`KVLayerCache` for autoregressive decoding.
        Only supported for self-attention (``context`` must be
        ``None``).

    Returns:
      ``output`` of shape ``(batch, seq_q, out_features)``, or
      ``(output, updated_cache)`` when ``kv_cache`` is provided.

    Raises:
      ValueError: If ``kv_cache`` is set together with ``context``.
    """
    batch, seq_q, _ = x.shape

    query = self.query_proj(x)
    if context is not None:
      key = self.key_proj(context)
      value = self.value_proj(context)
      seq_kv = context.shape[1]
    else:
      key = self.key_proj(x)
      value = self.value_proj(x)
      seq_kv = seq_q

    if kv_cache is not None and context is not None:
      raise ValueError(
        "kv_cache is only supported for self-attention (context must be None)."
      )

    head_dim = self.qkv_features // self.num_heads
    query = query.reshape(batch, seq_q, self.num_heads, head_dim)
    query = jnp.transpose(query, (0, 2, 1, 3))

    key = key.reshape(batch, seq_kv, self.num_kv_heads, head_dim)
    key = jnp.transpose(key, (0, 2, 1, 3))

    value = value.reshape(batch, seq_kv, self.num_kv_heads, head_dim)
    value = jnp.transpose(value, (0, 2, 1, 3))

    if self.use_rope:
      if position_ids_q is None:
        pos_q = jnp.arange(seq_q)[None, :].astype(jnp.int32)
        pos_q = jnp.broadcast_to(pos_q, (batch, seq_q))
      else:
        pos_q = position_ids_q
      cos_q, sin_q = rope_cos_sin_from_positions(
        pos_q,
        head_dim,
        self.rope_base,
        self.rope_max_positions,
        out_dtype=query.dtype,
      )
      cos_q = cos_q[:, None, :, :]
      sin_q = sin_q[:, None, :, :]
      query = apply_rope(query, cos_q, sin_q)

      if position_ids_kv is None:
        pos_kv = jnp.arange(seq_kv)[None, :].astype(jnp.int32)
        pos_kv = jnp.broadcast_to(pos_kv, (batch, seq_kv))
      else:
        pos_kv = position_ids_kv
      cos_k, sin_k = rope_cos_sin_from_positions(
        pos_kv,
        head_dim,
        self.rope_base,
        self.rope_max_positions,
        out_dtype=key.dtype,
      )
      cos_k = cos_k[:, None, :, :]
      sin_k = sin_k[:, None, :, :]
      key = apply_rope(key, cos_k, sin_k)

    key_kv = key
    value_kv = value

    if kv_cache is not None:
      past_len = int(kv_cache.length)
      k_hist = kv_cache.keys[:, :, :past_len, :]
      v_hist = kv_cache.values[:, :, :past_len, :]
      key_cat = jnp.concatenate([k_hist, key_kv], axis=2)
      value_cat = jnp.concatenate([v_hist, value_kv], axis=2)
      new_kv_cache = update_kv_layer_cache(
        kv_cache, key_kv, value_kv, past_len
      )
      key = _repeat_kv_heads(
        key_cat, num_kv_heads=self.num_kv_heads, num_heads=self.num_heads
      )
      value = _repeat_kv_heads(
        value_cat, num_kv_heads=self.num_kv_heads, num_heads=self.num_heads
      )
    else:
      key = _repeat_kv_heads(
        key_kv, num_kv_heads=self.num_kv_heads, num_heads=self.num_heads
      )
      value = _repeat_kv_heads(
        value_kv, num_kv_heads=self.num_kv_heads, num_heads=self.num_heads
      )
      new_kv_cache = None

    # Also materialise the sliding window as a boolean mask so backends
    # that ignore score_mod (e.g. cuDNN flash) still respect the window.
    if (
      self.attention_window is not None
      and context is None
    ):
      seq_kv_eff = key.shape[2]
      window_mask = make_sliding_window_mask(
        seq_q=seq_q,
        seq_kv=seq_kv_eff,
        window_size=self.attention_window,
        causal=True,
      )
      if kv_cache is not None:
        # Shift query positions by past_len so the window stays causal.
        past_len = int(kv_cache.length)
        q_idx = jnp.arange(seq_q) + past_len
        k_idx = jnp.arange(seq_kv_eff)
        window_mask = (
          (k_idx[None, :] <= q_idx[:, None])
          & (q_idx[:, None] - k_idx[None, :] < self.attention_window)
        )[None, None, :, :]
      mask = window_mask if mask is None else (mask & window_mask)

    effective_score_mod = compose_score_mods(self._base_score_mod, score_mod)

    output = self._attention_fn(
      query,
      key,
      value,
      mask=mask,
      score_mod=effective_score_mod,
      dropout_rng=None,
      dropout_rate=self.dropout_rate,
      deterministic=deterministic if deterministic is not None else True,
    )

    output = jnp.transpose(output, (0, 2, 1, 3))
    output = output.reshape(batch, seq_q, self.qkv_features)

    output = self.output_proj(output)
    output = self.dropout(output, deterministic=deterministic)

    if kv_cache is not None:
      return output, new_kv_cache
    return output
