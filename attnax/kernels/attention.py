# SPDX-License-Identifier: Apache-2.0

"""Pure-JAX attention kernels."""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from ._api import ScoreMod, _apply_score_mod


def standard_attention(
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
  """Scaled dot-product attention.

  .. math::

     \\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\!\\left(
       \\frac{QK^\\top}{\\sqrt{d_k}} + \\Delta\\right) V

  where :math:`\\Delta` is the optional ``score_mod`` bias. Entries
  with ``mask == False`` are set to :math:`-\\infty` before the
  softmax. Activation memory is :math:`O(n^2)`.

  Args:
    query: Array of shape ``(batch, num_heads, seq_q, head_dim)``.
    key: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    value: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    mask: Boolean mask broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``. ``True`` means attend.
    score_mod: Callable applied to the pre-softmax scores. See
      :mod:`attnax.kernels.score_mods`.
    dropout_rng: PRNG key for attention-weight dropout.
    dropout_rate: Dropout probability applied to attention weights.
    deterministic: If ``True``, disables dropout.

  Returns:
    Array of shape ``(batch, num_heads, seq_q, head_dim)``.
  """
  depth = query.shape[-1]
  scale = jax.lax.rsqrt(jnp.asarray(depth, dtype=query.dtype))
  scores = jnp.einsum("bhqd,bhkd->bhqk", query, key) * scale

  scores = _apply_score_mod(scores, score_mod)

  if mask is not None:
    large_neg = jnp.finfo(scores.dtype).min
    scores = jnp.where(mask, scores, large_neg)

  attn_weights = jax.nn.softmax(scores, axis=-1)

  if not deterministic and dropout_rate > 0.0 and dropout_rng is not None:
    keep_prob = 1.0 - dropout_rate
    keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    attn_weights = jnp.where(keep, attn_weights / keep_prob, 0.0)

  return jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value)


def memory_efficient_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  *,
  mask: Optional[jnp.ndarray] = None,
  score_mod: Optional[ScoreMod] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
  block_size: int = 512,
) -> jnp.ndarray:
  """Block-wise attention with :math:`O(n)` activation memory.

  Tiles the key/value sequence into blocks of size ``block_size`` and
  accumulates a running max, softmax denominator, and output via the
  online-softmax recurrence. Mathematically identical to
  :func:`standard_attention` but does not materialise the full
  ``(seq_q, seq_kv)`` score matrix. Falls back to
  :func:`standard_attention` when both axes fit in one block.

  Args:
    query: Array of shape ``(batch, num_heads, seq_q, head_dim)``.
    key: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    value: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    mask: Boolean mask broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``.
    score_mod: Callable applied block-by-block to the pre-softmax
      scores; position indices are global.
    dropout_rng: PRNG key for output dropout.
    dropout_rate: Dropout probability applied to the output.
    deterministic: If ``True``, disables dropout.
    block_size: Number of positions per query / key block.

  Returns:
    Array of shape ``(batch, num_heads, seq_q, head_dim)``.

  References:
    Milakov and Gimelshein, `Online normalizer calculation for softmax
    <https://arxiv.org/abs/1805.02867>`_, 2018.

    Dao et al., `FlashAttention: Fast and Memory-Efficient Exact
    Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`_,
    2022.
  """
  batch, num_heads, seq_q, head_dim = query.shape
  _, _, seq_kv, _ = key.shape

  if seq_q <= block_size and seq_kv <= block_size:
    return standard_attention(
      query,
      key,
      value,
      mask=mask,
      score_mod=score_mod,
      dropout_rng=dropout_rng,
      dropout_rate=dropout_rate,
      deterministic=deterministic,
    )

  scale = jax.lax.rsqrt(jnp.asarray(head_dim, dtype=query.dtype))
  large_neg_scalar = jnp.finfo(query.dtype).min

  num_q_blocks = (seq_q + block_size - 1) // block_size
  num_kv_blocks = (seq_kv + block_size - 1) // block_size

  b_idx_full = jnp.arange(batch, dtype=jnp.int32)[:, None, None, None]
  h_idx_full = jnp.arange(num_heads, dtype=jnp.int32)[None, :, None, None]

  output = jnp.zeros_like(query)

  for q_block in range(num_q_blocks):
    q_start = q_block * block_size
    q_end = min(q_start + block_size, seq_q)
    bq = q_end - q_start
    query_block = query[:, :, q_start:q_end, :]

    m_running = jnp.full(
      (batch, num_heads, bq, 1), large_neg_scalar, dtype=query.dtype
    )
    l_running = jnp.zeros((batch, num_heads, bq, 1), dtype=query.dtype)
    o_running = jnp.zeros((batch, num_heads, bq, head_dim), dtype=query.dtype)

    for kv_block in range(num_kv_blocks):
      kv_start = kv_block * block_size
      kv_end = min(kv_start + block_size, seq_kv)

      key_block = key[:, :, kv_start:kv_end, :]
      value_block = value[:, :, kv_start:kv_end, :]

      scores = (
        jnp.einsum("bhqd,bhkd->bhqk", query_block, key_block) * scale
      )

      if score_mod is not None:
        q_idx_block = jnp.arange(q_start, q_end, dtype=jnp.int32)
        kv_idx_block = jnp.arange(kv_start, kv_end, dtype=jnp.int32)
        scores = score_mod(
          scores,
          b_idx_full,
          h_idx_full,
          q_idx_block[None, None, :, None],
          kv_idx_block[None, None, None, :],
        )

      if mask is not None:
        mask_block = mask[..., q_start:q_end, kv_start:kv_end]
        scores = jnp.where(mask_block, scores, large_neg_scalar)

      # Online-softmax recurrence: rescale the accumulator by
      # exp(m_running - m_new) before adding the new block's
      # contribution. See Dao et al. (2022), Algorithm 1.
      m_block = jnp.max(scores, axis=-1, keepdims=True)
      m_new = jnp.maximum(m_running, m_block)
      alpha = jnp.exp(m_running - m_new)
      p_block = jnp.exp(scores - m_new)

      l_running = l_running * alpha + jnp.sum(p_block, axis=-1, keepdims=True)
      o_running = (
        o_running * alpha
        + jnp.einsum("bhqk,bhkd->bhqd", p_block, value_block)
      )
      m_running = m_new

    block_output = o_running / (l_running + 1e-10)
    output = output.at[:, :, q_start:q_end, :].set(block_output)

  if not deterministic and dropout_rate > 0.0 and dropout_rng is not None:
    keep_prob = 1.0 - dropout_rate
    keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
    output = jnp.where(keep, output / keep_prob, 0.0)

  return output


def flash_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  *,
  mask: Optional[jnp.ndarray] = None,
  score_mod: Optional[ScoreMod] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
  block_size: int = 512,
) -> jnp.ndarray:
  """Hardware-dispatched scaled dot-product attention.

  Dispatches to :func:`jax.nn.dot_product_attention` on GPU (backed by
  cuDNN's FlashAttention when available) and to
  :func:`memory_efficient_attention` on other backends. When
  ``score_mod`` is set, always uses
  :func:`memory_efficient_attention`.

  Args:
    query: Array of shape ``(batch, num_heads, seq_q, head_dim)``.
    key: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    value: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    mask: Boolean mask broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``.
    score_mod: Callable applied to the pre-softmax scores. Forces the
      memory-efficient fallback when set.
    dropout_rng: PRNG key for dropout (fallback path only).
    dropout_rate: Dropout probability.
    deterministic: If ``True``, disables dropout.
    block_size: Block size for the fallback path.

  Returns:
    Array of shape ``(batch, num_heads, seq_q, head_dim)``.

  References:
    Dao et al., `FlashAttention: Fast and Memory-Efficient Exact
    Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`_,
    2022.
  """
  backend = jax.default_backend()

  if (
    score_mod is None
    and backend == "gpu"
    and hasattr(jax.nn, "dot_product_attention")
  ):
    batch, num_heads, seq_q, head_dim = query.shape
    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    if mask is not None:
      mask = jnp.broadcast_to(
        mask, (batch, num_heads, seq_q, key.shape[2])
      )

    scale = 1.0 / math.sqrt(float(head_dim))
    output = jax.nn.dot_product_attention(
      q, k, v, bias=None, mask=mask, scale=scale
    )
    return jnp.transpose(output, (0, 2, 1, 3))

  return memory_efficient_attention(
    query,
    key,
    value,
    mask=mask,
    score_mod=score_mod,
    dropout_rng=dropout_rng,
    dropout_rate=dropout_rate,
    deterministic=deterministic,
    block_size=block_size,
  )


def lite_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  gate_proj: nnx.Linear,
  *,
  mask: Optional[jnp.ndarray] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
) -> jnp.ndarray:
  """Element-wise gated attention.

  Replaces the :math:`QK^\\top` matmul with the Hadamard product
  :math:`Q \\odot K` and a learnable linear gate. Carries trainable
  state (``gate_proj``) and therefore does not conform to the
  :data:`AttentionFn` protocol; selected through
  :attr:`~attnax.AttentionType.LITE`.

  Args:
    query: Array of shape ``(batch, num_heads, seq_q, head_dim)``.
    key: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    value: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    gate_proj: Linear layer mapping ``(head_dim,)`` to a scalar gate
      logit.
    mask: Boolean mask broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``.
    dropout_rng: PRNG key for output dropout.
    dropout_rate: Dropout probability.
    deterministic: If ``True``, disables dropout.

  Returns:
    Array of shape ``(batch, num_heads, seq_q, head_dim)``.
  """
  batch, num_heads, seq_q, head_dim = query.shape
  _, _, seq_kv, _ = key.shape

  if seq_q == seq_kv:
    attention_scores = query * key
  else:
    key_expanded = jnp.repeat(key[:, :, :1, :], seq_q, axis=2)
    attention_scores = query * key_expanded

  scores_flat = attention_scores.reshape(-1, head_dim)
  gate_scores = gate_proj(scores_flat)
  gate_scores = gate_scores.reshape(batch, num_heads, seq_q, 1)

  attn_weights = jax.nn.softmax(gate_scores, axis=-2)

  if mask is not None:
    mask_expanded = mask[..., :1]
    attn_weights = attn_weights * mask_expanded

  if seq_q == seq_kv:
    output = attn_weights * value
  else:
    value_expanded = jnp.repeat(value[:, :, :1, :], seq_q, axis=2)
    output = attn_weights * value_expanded

  if not deterministic and dropout_rate > 0.0 and dropout_rng is not None:
    keep_prob = 1.0 - dropout_rate
    keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
    output = jnp.where(keep, output / keep_prob, 0.0)

  return output


def _phi(x: jnp.ndarray) -> jnp.ndarray:
  """Feature map :math:`\\phi(x) = \\mathrm{elu}(x) + 1`."""
  return jax.nn.elu(x) + 1.0


def _linear_attention_non_causal(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  *,
  mask: Optional[jnp.ndarray],
  dropout_rng: Optional[jax.Array],
  dropout_rate: float,
  deterministic: bool,
) -> jnp.ndarray:
  """Non-causal linear attention as a single matmul."""
  q_phi = _phi(query)
  k_phi = _phi(key)
  if mask is not None:
    batch, num_heads, seq_q, _ = query.shape
    mask = jnp.broadcast_to(mask, (batch, num_heads, seq_q, seq_q))
    key_mask = mask[:, :, 0, :, None]
    k_phi = jnp.where(key_mask, k_phi, 0.0)
    value = jnp.where(key_mask, value, 0.0)
  s = jnp.einsum("bhkd,bhke->bhde", k_phi, value)
  z = jnp.sum(k_phi, axis=2)
  numer = jnp.einsum("bhqd,bhde->bhqe", q_phi, s)
  denom = jnp.einsum("bhqd,bhd->bhq", q_phi, z)
  out = numer / (denom[..., None] + 1e-6)
  if not deterministic and dropout_rate > 0.0 and dropout_rng is not None:
    keep_prob = 1.0 - dropout_rate
    keep = jax.random.bernoulli(dropout_rng, keep_prob, out.shape)
    out = jnp.where(keep, out / keep_prob, 0.0)
  return out


def linear_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  *,
  mask: Optional[jnp.ndarray] = None,
  score_mod: Optional[ScoreMod] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
  causal: bool = True,
  chunk_size: int = 64,
) -> jnp.ndarray:
  """Chunkwise-parallel linear attention.

  Softmax-free attention with the recurrence

  .. math::

     S_t = S_{t-1} + \\phi(K_t)^\\top V_t,
     \\qquad
     Y_t = \\frac{\\phi(Q_t)\\,S_t}{\\phi(Q_t)\\,Z_t + \\epsilon},

  where :math:`\\phi(x) = \\mathrm{elu}(x) + 1` and :math:`Z_t` is the
  running sum of :math:`\\phi(K)`. Inside a chunk, attention is a
  softmax-free matmul; across chunks, a ``(head_dim, head_dim)`` state
  and a ``(head_dim,)`` normaliser are propagated with
  :func:`jax.lax.scan`. Activation memory is :math:`O(L \\, d^2)`.

  Args:
    query: Array of shape ``(batch, num_heads, seq_q, head_dim)``.
    key: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
      ``seq_kv`` must equal ``seq_q``.
    value: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    mask: Boolean mask broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``. Only the first row along
      the query axis is consulted (treated as a key-padding mask).
    score_mod: Must be ``None``.
    dropout_rng: PRNG key for output dropout.
    dropout_rate: Dropout probability applied to the output.
    deterministic: If ``True``, disables dropout.
    causal: If ``True``, intra-chunk attention is causal and
      inter-chunk state propagates left-to-right.
    chunk_size: Tokens per chunk; ``seq_q`` must be divisible by
      ``chunk_size``.

  Returns:
    Array of shape ``(batch, num_heads, seq_q, head_dim)``.

  Raises:
    NotImplementedError: If ``score_mod`` is set.
    ValueError: If ``seq_q != seq_kv`` or ``seq_q`` is not divisible
      by ``chunk_size``.

  References:
    Katharopoulos et al., `Transformers are RNNs: Fast Autoregressive
    Transformers with Linear Attention
    <https://arxiv.org/abs/2006.16236>`_, 2020.
  """
  if score_mod is not None:
    raise NotImplementedError(
      "linear_attention is softmax-free and has no scores to bias; "
      "use standard_attention / memory_efficient_attention with the "
      "desired score_mod, or write a custom AttentionFn."
    )

  batch, num_heads, seq_q, head_dim = query.shape
  if key.shape[2] != seq_q or value.shape[2] != seq_q:
    raise ValueError(
      "linear_attention requires self-attention with matching Q/K/V "
      f"sequence lengths; got Q={seq_q}, K={key.shape[2]}, "
      f"V={value.shape[2]}"
    )
  if not causal:
    return _linear_attention_non_causal(
      query, key, value, mask=mask,
      dropout_rng=dropout_rng, dropout_rate=dropout_rate,
      deterministic=deterministic,
    )
  if seq_q % chunk_size != 0:
    raise ValueError(
      f"seq_q ({seq_q}) must be divisible by chunk_size "
      f"({chunk_size}); pad the sequence to a multiple of chunk_size"
    )

  q_phi = _phi(query)
  k_phi = _phi(key)

  if mask is not None:
    mask = jnp.broadcast_to(mask, (batch, num_heads, seq_q, seq_q))
    key_mask = mask[:, :, 0, :, None]
    k_phi = jnp.where(key_mask, k_phi, 0.0)
    value = jnp.where(key_mask, value, 0.0)

  num_chunks = seq_q // chunk_size
  q_chunks = q_phi.reshape(
    batch, num_heads, num_chunks, chunk_size, head_dim
  )
  k_chunks = k_phi.reshape(
    batch, num_heads, num_chunks, chunk_size, head_dim
  )
  v_chunks = value.reshape(
    batch, num_heads, num_chunks, chunk_size, head_dim
  )

  positions = jnp.arange(chunk_size)
  causal_intra = (
    positions[:, None] >= positions[None, :]
  ).astype(query.dtype)

  def step(state, inputs):
    s, z = state
    q_c, k_c, v_c = inputs

    inter_out = jnp.einsum("bhcd,bhde->bhce", q_c, s)
    inter_z = jnp.einsum("bhcd,bhd->bhc", q_c, z)

    qk = jnp.einsum("bhcd,bhed->bhce", q_c, k_c) * causal_intra
    intra_out = jnp.einsum("bhce,bhed->bhcd", qk, v_c)
    intra_z = jnp.sum(qk, axis=-1)

    out_chunk = (inter_out + intra_out) / (
      inter_z[..., None] + intra_z[..., None] + 1e-6
    )

    s_next = s + jnp.einsum("bhcd,bhce->bhde", k_c, v_c)
    z_next = z + jnp.sum(k_c, axis=2)
    return (s_next, z_next), out_chunk

  init_state = (
    jnp.zeros((batch, num_heads, head_dim, head_dim), dtype=query.dtype),
    jnp.zeros((batch, num_heads, head_dim), dtype=query.dtype),
  )
  scan_inputs = (
    jnp.transpose(q_chunks, (2, 0, 1, 3, 4)),
    jnp.transpose(k_chunks, (2, 0, 1, 3, 4)),
    jnp.transpose(v_chunks, (2, 0, 1, 3, 4)),
  )
  _, out_per_chunk = jax.lax.scan(step, init_state, scan_inputs)
  out = jnp.transpose(out_per_chunk, (1, 2, 0, 3, 4)).reshape(
    batch, num_heads, seq_q, head_dim
  )

  if not deterministic and dropout_rate > 0.0 and dropout_rng is not None:
    keep_prob = 1.0 - dropout_rate
    keep = jax.random.bernoulli(dropout_rng, keep_prob, out.shape)
    out = jnp.where(keep, out / keep_prob, 0.0)

  return out


def ring_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  *,
  mask: Optional[jnp.ndarray] = None,
  score_mod: Optional[ScoreMod] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
  axis_name: Optional[str] = None,
  block_size: int = 512,
) -> jnp.ndarray:
  """Sequence-parallel ring attention.

  Each device holds a shard of ``Q``, ``K``, ``V`` along the sequence
  axis. The kernel rotates the local ``K`` / ``V`` shard around a ring
  of devices via :func:`jax.lax.ppermute`, accumulating online-softmax
  statistics so that each device has attended to every key/value
  after one full rotation. Activation memory is :math:`O(L / P)` per
  device for sequence length :math:`L` across :math:`P` devices.
  Falls back to :func:`memory_efficient_attention` when ``axis_name``
  is ``None`` or the named axis is unbound.

  Args:
    query: Local shard of shape
      ``(batch, num_heads, seq_q_local, head_dim)``.
    key: Local shard of shape
      ``(batch, num_heads, seq_kv_local, head_dim)``.
    value: Local shard of shape
      ``(batch, num_heads, seq_kv_local, head_dim)``.
    mask: Boolean mask broadcastable to the local score shape
      ``(batch, num_heads, seq_q_local, seq_kv_local)``.
    score_mod: Callable applied to the pre-softmax scores; position
      indices are global.
    dropout_rng: PRNG key for output dropout.
    dropout_rate: Output dropout probability.
    deterministic: If ``True``, disables dropout.
    axis_name: Mesh axis to ring-permute over. When ``None``, runs on
      the current device.
    block_size: Block size for the single-device fallback.

  Returns:
    Local output shard of shape
    ``(batch, num_heads, seq_q_local, head_dim)``.

  References:
    Liu et al., `Ring Attention with Blockwise Transformers for
    Near-Infinite Context <https://arxiv.org/abs/2310.01889>`_, 2023.
  """
  if axis_name is None:
    return memory_efficient_attention(
      query,
      key,
      value,
      mask=mask,
      score_mod=score_mod,
      dropout_rng=dropout_rng,
      dropout_rate=dropout_rate,
      deterministic=deterministic,
      block_size=block_size,
    )

  try:
    axis_size = jax.lax.axis_size(axis_name)
  except Exception:
    return memory_efficient_attention(
      query,
      key,
      value,
      mask=mask,
      score_mod=score_mod,
      dropout_rng=dropout_rng,
      dropout_rate=dropout_rate,
      deterministic=deterministic,
      block_size=block_size,
    )
  axis_idx = jax.lax.axis_index(axis_name)

  batch, num_heads, seq_q_local, head_dim = query.shape
  seq_kv_local = key.shape[2]

  scale = jax.lax.rsqrt(jnp.asarray(head_dim, dtype=query.dtype))
  large_neg = jnp.finfo(query.dtype).min

  m_running = jnp.full(
    (batch, num_heads, seq_q_local, 1), large_neg, dtype=query.dtype
  )
  l_running = jnp.zeros(
    (batch, num_heads, seq_q_local, 1), dtype=query.dtype
  )
  o_running = jnp.zeros_like(query)

  k_cur = key
  v_cur = value
  perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]

  def step(carry, i):
    m_running, l_running, o_running, k_cur, v_cur = carry

    kv_owner = (axis_idx - i) % axis_size
    q_global_start = axis_idx * seq_q_local
    kv_global_start = kv_owner * seq_kv_local

    scores = jnp.einsum("bhqd,bhkd->bhqk", query, k_cur) * scale

    if score_mod is not None:
      b_idx = jnp.arange(batch, dtype=jnp.int32)[:, None, None, None]
      h_idx = jnp.arange(
        num_heads, dtype=jnp.int32
      )[None, :, None, None]
      q_idx = (
        jnp.arange(seq_q_local, dtype=jnp.int32) + q_global_start
      )[None, None, :, None]
      kv_idx = (
        jnp.arange(seq_kv_local, dtype=jnp.int32) + kv_global_start
      )[None, None, None, :]
      scores = score_mod(scores, b_idx, h_idx, q_idx, kv_idx)

    if mask is not None:
      scores = jnp.where(mask, scores, large_neg)

    m_block = jnp.max(scores, axis=-1, keepdims=True)
    m_new = jnp.maximum(m_running, m_block)
    alpha = jnp.exp(m_running - m_new)
    p_block = jnp.exp(scores - m_new)

    l_new = l_running * alpha + jnp.sum(p_block, axis=-1, keepdims=True)
    o_new = (
      o_running * alpha
      + jnp.einsum("bhqk,bhkd->bhqd", p_block, v_cur)
    )

    k_next = jax.lax.ppermute(k_cur, axis_name=axis_name, perm=perm)
    v_next = jax.lax.ppermute(v_cur, axis_name=axis_name, perm=perm)

    return (m_new, l_new, o_new, k_next, v_next), None

  (m_running, l_running, o_running, _, _), _ = jax.lax.scan(
    step,
    (m_running, l_running, o_running, k_cur, v_cur),
    jnp.arange(axis_size),
  )

  output = o_running / (l_running + 1e-10)

  if not deterministic and dropout_rate > 0.0 and dropout_rng is not None:
    keep_prob = 1.0 - dropout_rate
    keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
    output = jnp.where(keep, output / keep_prob, 0.0)

  return output


def ring_attention_reference(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  *,
  mask: Optional[jnp.ndarray] = None,
  score_mod: Optional[ScoreMod] = None,
) -> jnp.ndarray:
  """Single-device reference for :func:`ring_attention`.

  Computes the same output as :func:`ring_attention` when the entire
  sequence resides on one device. Used for numerical equivalence
  tests.

  Args:
    query: Array of shape ``(batch, num_heads, seq_q, head_dim)``.
    key: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    value: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    mask: Boolean mask broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``.
    score_mod: Callable applied to the pre-softmax scores.

  Returns:
    Array of shape ``(batch, num_heads, seq_q, head_dim)``.
  """
  return standard_attention(
    query, key, value, mask=mask, score_mod=score_mod
  )
