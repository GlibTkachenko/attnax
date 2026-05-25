# SPDX-License-Identifier: Apache-2.0

"""Pallas FlashAttention kernel."""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp

from ._api import ScoreMod
from .attention import memory_efficient_attention


def _supports_pallas() -> bool:
  """Returns whether Pallas is importable on a GPU/TPU backend."""
  try:
    import jax.experimental.pallas as _pl  # noqa: F401
  except ImportError:
    return False
  backend = jax.default_backend()
  return backend in ("gpu", "tpu")


def pallas_flash_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  *,
  mask: Optional[jnp.ndarray] = None,
  score_mod: Optional[ScoreMod] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
  block_q: int = 128,
  block_kv: int = 128,
  force_fallback: bool = False,
) -> jnp.ndarray:
  """FlashAttention forward pass compiled with Pallas.

  Dispatches to a Pallas-lowered kernel on GPU/TPU and falls back to
  :func:`~attnax.kernels.memory_efficient_attention` on CPU or when
  the kernel fails to lower. Forward pass only; the gradient is
  computed by tracing the kernel with ``jvp`` rather than by a
  hand-written backward.

  Args:
    query: Array of shape ``(batch, num_heads, seq_q, head_dim)``.
    key: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    value: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
    mask: Boolean mask broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``.
    score_mod: Callable traced into the inner loop; position indices
      are global.
    dropout_rng: PRNG key for output dropout.
    dropout_rate: Output dropout probability.
    deterministic: If ``True``, disables dropout.
    block_q: Queries per kernel program. ``seq_q`` must be divisible
      by ``block_q``.
    block_kv: Keys per inner-loop block. ``seq_kv`` must be divisible
      by ``block_kv``.
    force_fallback: If ``True``, always use the pure-JAX fallback.

  Returns:
    Array of shape ``(batch, num_heads, seq_q, head_dim)``.

  References:
    Dao et al., `FlashAttention: Fast and Memory-Efficient Exact
    Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`_,
    2022.
  """
  if force_fallback or not _supports_pallas():
    return memory_efficient_attention(
      query,
      key,
      value,
      mask=mask,
      score_mod=score_mod,
      dropout_rng=dropout_rng,
      dropout_rate=dropout_rate,
      deterministic=deterministic,
      block_size=max(block_q, block_kv),
    )

  try:
    output = _pallas_forward(
      query, key, value,
      mask=mask,
      score_mod=score_mod,
      block_q=block_q,
      block_kv=block_kv,
    )
  except Exception:
    # Any Pallas lowering failure (unsupported head_dim, unsupported
    # dtype, missing toolchain, …) degrades to the pure-JAX path.
    return memory_efficient_attention(
      query,
      key,
      value,
      mask=mask,
      score_mod=score_mod,
      dropout_rng=dropout_rng,
      dropout_rate=dropout_rate,
      deterministic=deterministic,
      block_size=max(block_q, block_kv),
    )

  if not deterministic and dropout_rate > 0.0 and dropout_rng is not None:
    keep_prob = 1.0 - dropout_rate
    keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
    output = jnp.where(keep, output / keep_prob, 0.0)

  return output


def _pallas_forward(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  *,
  mask: Optional[jnp.ndarray],
  score_mod: Optional[ScoreMod],
  block_q: int,
  block_kv: int,
) -> jnp.ndarray:
  """Compiles and dispatches the Pallas FlashAttention kernel."""
  import jax.experimental.pallas as pl

  batch, num_heads, seq_q, head_dim = query.shape
  _, _, seq_kv, _ = key.shape
  scale = 1.0 / math.sqrt(float(head_dim))

  if seq_q % block_q != 0:
    raise ValueError(
      f"seq_q ({seq_q}) must be divisible by block_q ({block_q}); pad "
      "the sequence or pick a different block_q."
    )
  if seq_kv % block_kv != 0:
    raise ValueError(
      f"seq_kv ({seq_kv}) must be divisible by block_kv ({block_kv}); "
      "pad the sequence or pick a different block_kv."
    )

  num_kv_blocks = seq_kv // block_kv

  def kernel(q_ref, k_ref, v_ref, mask_ref, o_ref):
    program_b = pl.program_id(0)
    program_h = pl.program_id(1)
    program_q = pl.program_id(2)
    q_block = q_ref[...]
    q_start = program_q * block_q

    large_neg = jnp.finfo(q_block.dtype).min
    m_i = jnp.full((block_q,), large_neg, dtype=jnp.float32)
    l_i = jnp.zeros((block_q,), dtype=jnp.float32)
    acc = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

    def body(i, carry):
      m_i, l_i, acc = carry
      k_block = pl.load(
        k_ref, (pl.ds(i * block_kv, block_kv), slice(None))
      )
      v_block = pl.load(
        v_ref, (pl.ds(i * block_kv, block_kv), slice(None))
      )
      scores = jnp.dot(q_block, k_block.T) * scale

      if score_mod is not None:
        q_idx = (q_start + jnp.arange(block_q, dtype=jnp.int32))
        kv_idx = (i * block_kv + jnp.arange(block_kv, dtype=jnp.int32))
        scores_4d = scores[None, None, :, :]
        b_idx = jnp.asarray(program_b, dtype=jnp.int32)[
          None, None, None, None
        ]
        h_idx = jnp.asarray(program_h, dtype=jnp.int32)[
          None, None, None, None
        ]
        scores_4d = score_mod(
          scores_4d,
          b_idx,
          h_idx,
          q_idx[None, None, :, None],
          kv_idx[None, None, None, :],
        )
        scores = scores_4d[0, 0]

      if mask_ref is not None:
        m_block = pl.load(
          mask_ref,
          (
            pl.ds(0, block_q),
            pl.ds(i * block_kv, block_kv),
          ),
        )
        scores = jnp.where(m_block, scores, large_neg)

      m_block_max = jnp.max(scores, axis=-1)
      m_new = jnp.maximum(m_i, m_block_max)
      alpha = jnp.exp(m_i - m_new)
      p = jnp.exp(scores - m_new[:, None])
      l_new = l_i * alpha + jnp.sum(p, axis=-1)
      acc = acc * alpha[:, None] + jnp.dot(p, v_block).astype(acc.dtype)
      return m_new, l_new, acc

    _, l_final, acc_final = jax.lax.fori_loop(
      0, num_kv_blocks, body, (m_i, l_i, acc)
    )
    o_ref[...] = (acc_final / l_final[:, None]).astype(o_ref.dtype)

  out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)
  in_specs = [
    pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, q: (b, h, q, 0)),
    pl.BlockSpec((1, 1, seq_kv, head_dim), lambda b, h, q: (b, h, 0, 0)),
    pl.BlockSpec((1, 1, seq_kv, head_dim), lambda b, h, q: (b, h, 0, 0)),
  ]
  out_spec = pl.BlockSpec(
    (1, 1, block_q, head_dim), lambda b, h, q: (b, h, q, 0)
  )
  args = [query, key, value]
  if mask is not None:
    mask = jnp.broadcast_to(mask, (batch, num_heads, seq_q, seq_kv))
    in_specs.append(
      pl.BlockSpec(
        (1, 1, block_q, seq_kv), lambda b, h, q: (b, h, q, 0)
      )
    )
    args.append(mask)
  else:
    # The kernel signature must be stable, so we pad with a dummy
    # zero-sized tensor when no mask is supplied. Pallas tolerates
    # zero-sized BlockSpecs as long as the kernel never reads from
    # them; the inner ``if mask_ref is not None`` guards access.
    in_specs.append(None)
    args.append(None)

  return pl.pallas_call(
    kernel,
    grid=(batch, num_heads, seq_q // block_q),
    in_specs=in_specs,
    out_specs=out_spec,
    out_shape=out_shape,
  )(*args)
