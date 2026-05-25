"""Tests for advanced kernels: linear, ring, paged, Pallas flash, MoE."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import pytest

import attnax
from attnax.kernels.attention import _linear_attention_non_causal, _phi


@pytest.fixture
def small_qkv():
  rng = jax.random.key(0)
  keys = jax.random.split(rng, 3)
  q = jax.random.normal(keys[0], (2, 4, 32, 16))
  k = jax.random.normal(keys[1], (2, 4, 32, 16))
  v = jax.random.normal(keys[2], (2, 4, 32, 16))
  return q, k, v


class TestLinearAttention:
  def test_output_shape(self, small_qkv):
    q, k, v = small_qkv
    out = attnax.linear_attention(q, k, v, chunk_size=8)
    assert out.shape == q.shape

  def test_score_mod_rejected(self, small_qkv):
    q, k, v = small_qkv
    with pytest.raises(NotImplementedError):
      attnax.linear_attention(
        q, k, v, chunk_size=8, score_mod=attnax.causal_mod()
      )

  def test_chunk_independence(self, small_qkv):
    """Different chunk sizes must produce the same causal output."""
    q, k, v = small_qkv
    out_small = attnax.linear_attention(q, k, v, chunk_size=4)
    out_large = attnax.linear_attention(q, k, v, chunk_size=8)
    np.testing.assert_allclose(out_small, out_large, atol=1e-4, rtol=1e-4)

  def test_matches_reference_causal(self, small_qkv):
    """Chunkwise causal output equals the direct cumulative formula."""
    q, k, v = small_qkv
    out = attnax.linear_attention(q, k, v, chunk_size=8, causal=True)

    q_phi = _phi(q)
    k_phi = _phi(k)
    seq_q = q.shape[2]
    causal = jnp.tril(jnp.ones((seq_q, seq_q), dtype=q.dtype))
    qk = jnp.einsum("bhqd,bhkd->bhqk", q_phi, k_phi) * causal
    numer = jnp.einsum("bhqk,bhke->bhqe", qk, v)
    denom = jnp.sum(qk, axis=-1, keepdims=True)
    expected = numer / (denom + 1e-6)
    np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

  def test_non_causal_matches_global_matmul(self, small_qkv):
    q, k, v = small_qkv
    out = attnax.linear_attention(q, k, v, chunk_size=8, causal=False)
    expected = _linear_attention_non_causal(
      q, k, v, mask=None, dropout_rng=None,
      dropout_rate=0.0, deterministic=True,
    )
    np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

  def test_jit_compatibility(self, small_qkv):
    q, k, v = small_qkv
    jitted = jax.jit(
      lambda q, k, v: attnax.linear_attention(q, k, v, chunk_size=8)
    )
    out_jit = jitted(q, k, v)
    out_eager = attnax.linear_attention(q, k, v, chunk_size=8)
    np.testing.assert_allclose(out_jit, out_eager, atol=1e-5, rtol=1e-5)

  def test_seq_len_validation(self, small_qkv):
    q, k, v = small_qkv
    with pytest.raises(ValueError, match="divisible by chunk_size"):
      attnax.linear_attention(q, k, v, chunk_size=7)


class TestRingAttention:
  def test_single_device_matches_reference(self, small_qkv):
    q, k, v = small_qkv
    out_ring = attnax.ring_attention(q, k, v)
    out_ref = attnax.ring_attention_reference(q, k, v)
    np.testing.assert_allclose(out_ring, out_ref, atol=1e-5, rtol=1e-5)

  def test_with_score_mod(self, small_qkv):
    q, k, v = small_qkv
    causal = attnax.causal_mod()
    out_ring = attnax.ring_attention(q, k, v, score_mod=causal)
    out_ref = attnax.ring_attention_reference(q, k, v, score_mod=causal)
    np.testing.assert_allclose(out_ring, out_ref, atol=1e-5, rtol=1e-5)

  def test_alibi_score_mod(self, small_qkv):
    """ALiBi via score_mod produces the documented bias."""
    q, k, v = small_qkv
    num_heads = q.shape[1]
    alibi = attnax.alibi_mod(num_heads)
    out_ring = attnax.ring_attention(q, k, v, score_mod=alibi)
    out_ref = attnax.ring_attention_reference(q, k, v, score_mod=alibi)
    np.testing.assert_allclose(out_ring, out_ref, atol=1e-5, rtol=1e-5)

  def test_jit_compatibility(self, small_qkv):
    q, k, v = small_qkv
    out_jit = jax.jit(attnax.ring_attention)(q, k, v)
    out_eager = attnax.ring_attention(q, k, v)
    np.testing.assert_allclose(out_jit, out_eager, atol=1e-5, rtol=1e-5)

  def test_unknown_axis_falls_back(self, small_qkv):
    """Passing an unbound axis_name silently falls back to local."""
    q, k, v = small_qkv
    out = attnax.ring_attention(q, k, v, axis_name="not_bound")
    out_ref = attnax.ring_attention_reference(q, k, v)
    np.testing.assert_allclose(out, out_ref, atol=1e-5, rtol=1e-5)


class TestPagedKVCache:
  def test_init_shapes(self):
    cache = attnax.init_paged_kv_cache(
      num_blocks=8, block_size=4, num_kv_heads=2, head_dim=16,
      batch_size=2, max_blocks_per_seq=4, dtype=jnp.float32,
    )
    assert cache.key_pool.shape == (8, 4, 2, 16)
    assert cache.value_pool.shape == (8, 4, 2, 16)
    assert cache.block_table.shape == (2, 4)
    assert cache.seq_lengths.shape == (2,)
    assert int(cache.seq_lengths[0]) == 0
    assert int(cache.block_table[0, 0]) == -1

  def test_allocate_and_append(self):
    cache = attnax.init_paged_kv_cache(
      num_blocks=8, block_size=4, num_kv_heads=2, head_dim=8,
      batch_size=1, max_blocks_per_seq=4, dtype=jnp.float32,
    )
    free = jnp.arange(8, dtype=jnp.int32)
    cache, used = attnax.allocate_blocks(
      cache, sequence_idx=0, num_new_tokens=10, free_block_ids=free,
    )
    assert used == 3  # 10 tokens / 4 per block -> 3 blocks
    assert int(cache.block_table[0, 0]) == 0
    assert int(cache.block_table[0, 1]) == 1
    assert int(cache.block_table[0, 2]) == 2

    rng = jax.random.key(0)
    keys_new = jax.random.normal(rng, (10, 2, 8))
    vals_new = jax.random.normal(rng, (10, 2, 8))
    cache = attnax.append_kv(
      cache, sequence_idx=0, keys_new=keys_new, values_new=vals_new
    )
    assert int(cache.seq_lengths[0]) == 10

    keys_back, vals_back, length = attnax.gather_kv(cache, sequence_idx=0)
    assert length == 10
    np.testing.assert_allclose(keys_back, keys_new)
    np.testing.assert_allclose(vals_back, vals_new)

  def test_append_without_alloc_raises(self):
    cache = attnax.init_paged_kv_cache(
      num_blocks=2, block_size=4, num_kv_heads=2, head_dim=8,
      batch_size=1, max_blocks_per_seq=2, dtype=jnp.float32,
    )
    with pytest.raises(ValueError, match="unallocated"):
      attnax.append_kv(
        cache, sequence_idx=0,
        keys_new=jnp.zeros((1, 2, 8)),
        values_new=jnp.zeros((1, 2, 8)),
      )

  def test_paged_attention_matches_standard(self):
    cache = attnax.init_paged_kv_cache(
      num_blocks=4, block_size=4, num_kv_heads=2, head_dim=8,
      batch_size=1, max_blocks_per_seq=4, dtype=jnp.float32,
    )
    free = jnp.arange(4, dtype=jnp.int32)
    cache, _ = attnax.allocate_blocks(
      cache, sequence_idx=0, num_new_tokens=12, free_block_ids=free,
    )
    rng = jax.random.key(0)
    keys = jax.random.split(rng, 3)
    keys_new = jax.random.normal(keys[0], (12, 2, 8))
    vals_new = jax.random.normal(keys[1], (12, 2, 8))
    cache = attnax.append_kv(
      cache, sequence_idx=0, keys_new=keys_new, values_new=vals_new
    )
    query = jax.random.normal(keys[2], (2, 3, 8))
    out_paged = attnax.paged_attention(query, cache, sequence_idx=0)

    q_full = query[None]  # (1, heads, seq_q, dim)
    k_full = jnp.transpose(keys_new, (1, 0, 2))[None]
    v_full = jnp.transpose(vals_new, (1, 0, 2))[None]
    out_std = attnax.standard_attention(q_full, k_full, v_full)[0]
    np.testing.assert_allclose(out_paged, out_std, atol=1e-5, rtol=1e-5)


class TestPallasFlashAttention:
  def test_cpu_fallback_matches_standard(self, small_qkv):
    q, k, v = small_qkv
    out_pf = attnax.pallas_flash_attention(q, k, v)
    out_std = attnax.standard_attention(q, k, v)
    np.testing.assert_allclose(out_pf, out_std, atol=1e-4, rtol=1e-4)

  def test_force_fallback(self, small_qkv):
    q, k, v = small_qkv
    out = attnax.pallas_flash_attention(q, k, v, force_fallback=True)
    out_ref = attnax.memory_efficient_attention(q, k, v, block_size=128)
    np.testing.assert_allclose(out, out_ref, atol=1e-5, rtol=1e-5)

  def test_with_score_mod(self, small_qkv):
    q, k, v = small_qkv
    causal = attnax.causal_mod()
    out_pf = attnax.pallas_flash_attention(
      q, k, v, score_mod=causal, force_fallback=True
    )
    out_std = attnax.standard_attention(q, k, v, score_mod=causal)
    np.testing.assert_allclose(out_pf, out_std, atol=1e-4, rtol=1e-4)

  def test_jit_compatibility(self, small_qkv):
    q, k, v = small_qkv
    out = jax.jit(attnax.pallas_flash_attention)(q, k, v)
    out_ref = attnax.standard_attention(q, k, v)
    np.testing.assert_allclose(out, out_ref, atol=1e-4, rtol=1e-4)


class TestMixtureOfExperts:
  def test_output_shape(self):
    moe = attnax.MixtureOfExperts(
      nnx.Rngs(0), d_model=32, d_ff=64, num_experts=4, top_k=2,
    )
    x = jnp.zeros((2, 8, 32))
    y, aux = moe(x, deterministic=True)
    assert y.shape == x.shape
    assert "load_balance_loss" in aux
    assert "router_entropy" in aux

  def test_invalid_top_k_rejected(self):
    with pytest.raises(ValueError, match="top_k"):
      attnax.MixtureOfExperts(
        nnx.Rngs(0), d_model=32, d_ff=64, num_experts=4, top_k=5,
      )

  def test_invalid_num_experts_rejected(self):
    with pytest.raises(ValueError, match="num_experts"):
      attnax.MixtureOfExperts(
        nnx.Rngs(0), d_model=32, d_ff=64, num_experts=0, top_k=1,
      )

  def test_jit_compatibility(self):
    moe = attnax.MixtureOfExperts(
      nnx.Rngs(0), d_model=32, d_ff=64, num_experts=4, top_k=2,
    )
    x = jnp.ones((2, 8, 32))

    @jax.jit
    def fwd(x):
      return moe(x, deterministic=True)

    y, aux = fwd(x)
    assert y.shape == x.shape

  def test_gradient_flow(self):
    moe = attnax.MixtureOfExperts(
      nnx.Rngs(0), d_model=16, d_ff=32, num_experts=4, top_k=2,
    )
    x = jnp.ones((1, 4, 16))

    def loss_fn(m):
      y, aux = m(x, deterministic=True)
      return jnp.sum(y**2) + aux["load_balance_loss"]

    grads = nnx.grad(loss_fn)(moe)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

  def test_activations(self):
    for act in ("swiglu", "geglu", "gelu", "relu"):
      moe = attnax.MixtureOfExperts(
        nnx.Rngs(0), d_model=16, d_ff=32, num_experts=4, top_k=2,
        ff_activation=act,
      )
      y, _ = moe(jnp.zeros((1, 4, 16)), deterministic=True)
      assert y.shape == (1, 4, 16)

  def test_load_balance_low_when_uniform(self):
    """Tied router weights produce a load-balance loss near 1."""
    moe = attnax.MixtureOfExperts(
      nnx.Rngs(0), d_model=16, d_ff=32, num_experts=4, top_k=2,
    )
    # Replace router with zeros so every expert is equally likely.
    moe.router = nnx.Param(jnp.zeros_like(moe.router[...]))
    rng = jax.random.key(7)
    x = jax.random.normal(rng, (4, 16, 16))
    _, aux = moe(x, deterministic=True)
    # With perfectly uniform routing, fraction*P_e sums to top_k/E so
    # the loss equals top_k. For top_k=2, E=4, loss ≈ 2.
    assert float(aux["load_balance_loss"]) < 2.5


class TestPluggableIntoMultiHeadAttention:
  """The new kernels should drop into MultiHeadAttention seamlessly."""

  def test_linear_as_attention_fn(self):
    attn = attnax.MultiHeadAttention(
      nnx.Rngs(0),
      num_heads=4,
      in_features=32,
      attention_fn=lambda q, k, v, **kw: attnax.linear_attention(
        q, k, v, chunk_size=4
      ),
    )
    x = jnp.ones((1, 16, 32))
    out = attn(x, deterministic=True)
    assert out.shape == x.shape

  def test_ring_as_attention_fn(self):
    attn = attnax.MultiHeadAttention(
      nnx.Rngs(0),
      num_heads=4,
      in_features=32,
      attention_fn=attnax.ring_attention,
    )
    x = jnp.ones((1, 16, 32))
    out = attn(x, deterministic=True)
    assert out.shape == x.shape

  def test_pallas_flash_as_attention_fn(self):
    attn = attnax.MultiHeadAttention(
      nnx.Rngs(0),
      num_heads=4,
      in_features=32,
      attention_fn=attnax.pallas_flash_attention,
    )
    x = jnp.ones((1, 16, 32))
    out = attn(x, deterministic=True)
    assert out.shape == x.shape


class TestAttentionTypeEnum:
  """The enum should cover every kernel that fits the AttentionFn shape."""

  def test_linear_enum_dispatches(self):
    attn = attnax.MultiHeadAttention(
      nnx.Rngs(0),
      num_heads=4,
      in_features=32,
      attention_type=attnax.AttentionType.LINEAR,
      linear_attention_chunk_size=4,
    )
    x = jnp.ones((1, 16, 32))
    out = attn(x, deterministic=True)
    assert out.shape == x.shape

  def test_pallas_flash_enum_dispatches(self):
    attn = attnax.MultiHeadAttention(
      nnx.Rngs(0),
      num_heads=4,
      in_features=32,
      attention_type=attnax.AttentionType.PALLAS_FLASH,
    )
    x = jnp.ones((1, 16, 32))
    out = attn(x, deterministic=True)
    assert out.shape == x.shape

  def test_linear_string_alias(self):
    """Enum values should accept their string form for JSON/YAML configs."""
    config = attnax.TransformerConfig(
      vocab_size=128,
      d_model=32,
      num_heads=4,
      num_layers=1,
      d_ff=64,
      max_len=16,
      attention_type="linear",
      linear_attention_chunk_size=4,
    )
    model = attnax.TransformerEncoder(nnx.Rngs(0), config)
    ids = jnp.ones((1, 16), dtype=jnp.int32)
    out = model(ids, deterministic=True)
    assert out.shape == (1, 16, 32)

  def test_pallas_flash_string_alias(self):
    config = attnax.TransformerConfig(
      vocab_size=128,
      d_model=32,
      num_heads=4,
      num_layers=1,
      d_ff=64,
      max_len=16,
      attention_type="pallas_flash",
    )
    model = attnax.TransformerEncoder(nnx.Rngs(0), config)
    ids = jnp.ones((1, 16), dtype=jnp.int32)
    out = model(ids, deterministic=True)
    assert out.shape == (1, 16, 32)
