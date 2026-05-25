# SPDX-License-Identifier: Apache-2.0

"""Tests for ``attnax.kernels``: backends, score-mods, strategy pattern."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from attnax import (
    AttentionType,
    MultiHeadAttention,
    TransformerConfig,
    TransformerEncoder,
    additive_bias_mod,
    alibi_mod,
    alibi_slopes,
    causal_mod,
    combine_masks,
    compose_score_mods,
    document_mask_mod,
    flash_attention,
    make_causal_mask,
    make_document_mask,
    make_sliding_window_mask,
    mask_mod_to_boolean_mask,
    mask_mod_to_score_mod,
    memory_efficient_attention,
    prefix_lm_mod,
    sliding_window_mod,
    standard_attention,
)


def _qkv(batch=2, heads=4, seq=8, dim=16, key=0):
    k = jax.random.key(key)
    k1, k2, k3 = jax.random.split(k, 3)
    q = jax.random.normal(k1, (batch, heads, seq, dim))
    kv = jax.random.normal(k2, (batch, heads, seq, dim))
    v = jax.random.normal(k3, (batch, heads, seq, dim))
    return q, kv, v


def test_standard_matches_memory_efficient_small():
    """memory_efficient must route to standard when seq <= block_size."""
    q, k, v = _qkv()
    out_std = standard_attention(q, k, v)
    out_mem = memory_efficient_attention(q, k, v, block_size=64)
    assert jnp.allclose(out_std, out_mem, atol=1e-5)


def test_memory_efficient_online_softmax_correct_for_long_seqs():
    """Online softmax must match standard attention when seq > block_size."""
    q, k, v = _qkv(batch=1, heads=2, seq=128, dim=8, key=42)
    out_std = standard_attention(q, k, v)
    out_mem = memory_efficient_attention(q, k, v, block_size=32)
    assert jnp.allclose(out_std, out_mem, atol=1e-4)


def test_zero_score_mod_is_identity():
    """A score_mod that adds zero must produce the same output as no mod."""
    q, k, v = _qkv()
    zero_mod = additive_bias_mod(jnp.zeros((1, 1, 1, 1)))
    out_a = standard_attention(q, k, v)
    out_b = standard_attention(q, k, v, score_mod=zero_mod)
    assert jnp.allclose(out_a, out_b, atol=1e-6)


def test_causal_mod_matches_causal_mask():
    """causal_mod must match the explicit causal boolean mask."""
    q, k, v = _qkv(seq=10)
    mask = make_causal_mask(10)
    out_mask = standard_attention(q, k, v, mask=mask)
    out_mod = standard_attention(q, k, v, score_mod=causal_mod())
    assert jnp.allclose(out_mask, out_mod, atol=1e-5)


def test_sliding_window_mod_matches_mask():
    """sliding_window_mod must equal the explicit sliding-window mask."""
    q, k, v = _qkv(seq=12)
    window = 4
    mask = make_sliding_window_mask(12, window_size=window, causal=True)
    out_mask = standard_attention(q, k, v, mask=mask)
    out_mod = standard_attention(
        q, k, v, score_mod=sliding_window_mod(window_size=window)
    )
    assert jnp.allclose(out_mask, out_mod, atol=1e-5)


def test_sliding_window_mask_shape_and_pattern():
    """make_sliding_window_mask returns the correct boolean pattern."""
    mask = make_sliding_window_mask(6, window_size=3, causal=True)
    assert mask.shape == (1, 1, 6, 6)
    # Row 5 should attend to keys [3, 4, 5].
    row5 = mask[0, 0, 5]
    assert bool(row5[3]) and bool(row5[4]) and bool(row5[5])
    assert not bool(row5[2])

    sym = make_sliding_window_mask(5, window_size=2, causal=False)
    assert bool(sym[0, 0, 2, 1]) and bool(sym[0, 0, 2, 3])
    assert not bool(sym[0, 0, 2, 4])


def test_make_sliding_window_mask_rejects_nonpositive():
    with pytest.raises(ValueError):
        make_sliding_window_mask(4, window_size=0)
    with pytest.raises(ValueError):
        make_sliding_window_mask(4, window_size=-1)


def test_alibi_slopes_shape_and_sign():
    s = alibi_slopes(8)
    assert s.shape == (8,)
    assert jnp.all(s > 0)
    # Slopes should be monotonically decreasing for power-of-two heads.
    assert jnp.all(s[1:] < s[:-1])


def test_alibi_slopes_non_power_of_two():
    """alibi_slopes returns the right shape for non-power-of-two head counts."""
    s = alibi_slopes(6)
    assert s.shape == (6,)
    assert jnp.all(s > 0)


def test_alibi_mod_adds_negative_distance_bias():
    """ALiBi must produce non-positive contributions equal to -m * |q-k|."""
    q, k, v = _qkv(heads=4, seq=8)
    zero = jnp.zeros((1, 4, 8, 8))
    biased = alibi_mod(num_heads=4)(
        zero,
        jnp.zeros((1, 1, 1, 1), dtype=jnp.int32),
        jnp.zeros((1, 4, 1, 1), dtype=jnp.int32),
        jnp.arange(8)[None, None, :, None].astype(jnp.int32),
        jnp.arange(8)[None, None, None, :].astype(jnp.int32),
    )
    assert biased.shape == (1, 4, 8, 8)
    # Diagonal should be exactly zero (distance 0).
    assert jnp.allclose(jnp.diagonal(biased[0, 0]), 0.0)
    # All off-diagonal entries should be < 0.
    assert jnp.all(biased[0, 0] <= 0)


def test_prefix_lm_mod_pattern():
    """prefix_lm_mod is bidirectional within the prefix and causal after."""
    seq, pref = 6, 3
    zero = jnp.zeros((1, 1, seq, seq))
    mod = prefix_lm_mod(jnp.array([pref]))
    out = mod(
        zero,
        jnp.zeros((1, 1, 1, 1), dtype=jnp.int32),
        jnp.zeros((1, 1, 1, 1), dtype=jnp.int32),
        jnp.arange(seq)[None, None, :, None].astype(jnp.int32),
        jnp.arange(seq)[None, None, None, :].astype(jnp.int32),
    )
    # Within prefix: all pairs (q < pref) attend everywhere within prefix.
    assert out[0, 0, 1, 2] == 0  # q=1, kv=2 both in prefix → unmasked
    # After prefix: causal only.
    assert out[0, 0, 3, 4] < 0  # q=3, kv=4 → masked (future)
    assert out[0, 0, 4, 3] == 0  # q=4, kv=3 → ok


def test_document_mask_isolates_documents():
    """make_document_mask + standard_attention must zero cross-doc attention."""
    doc_ids = jnp.array([[0, 0, 0, 1, 1]])
    mask = make_document_mask(doc_ids)
    assert mask.shape == (1, 1, 5, 5)
    # Tokens in different docs must be masked out.
    assert not bool(mask[0, 0, 0, 3])
    assert bool(mask[0, 0, 0, 1])


def test_document_mask_mod_matches_explicit_mask():
    """document_mask_mod must produce the same output as make_document_mask."""
    doc_ids = jnp.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 2]])
    q, k, v = _qkv(batch=2, heads=4, seq=5, dim=16, key=7)
    mask = make_document_mask(doc_ids)
    out_mask = standard_attention(q, k, v, mask=mask)
    out_mod = standard_attention(
        q, k, v, score_mod=document_mask_mod(doc_ids)
    )
    assert jnp.allclose(out_mask, out_mod, atol=1e-5)


def test_compose_score_mods_order():
    """compose_score_mods applies its arguments left-to-right."""
    q, k, v = _qkv(seq=8)
    combined = compose_score_mods(
        causal_mod(), sliding_window_mod(window_size=4)
    )
    out = standard_attention(q, k, v, score_mod=combined)
    # Equivalent explicit mask.
    causal = make_causal_mask(8)
    window = make_sliding_window_mask(8, window_size=4, causal=True)
    out_mask = standard_attention(q, k, v, mask=combine_masks(causal, window))
    assert jnp.allclose(out, out_mask, atol=1e-5)


def test_compose_score_mods_none_passthrough():
    """compose_score_mods returns None when all arguments are None."""
    assert compose_score_mods(None, None) is None
    mod = causal_mod()
    assert compose_score_mods(None, mod) is mod


def test_mask_mod_to_boolean_mask_round_trip():
    """mask_mod_to_boolean_mask should materialise the bool pattern."""

    def causal(b, h, q, kv):
        return kv <= q

    mask = mask_mod_to_boolean_mask(
        causal, batch=1, num_heads=2, seq_q=4, seq_kv=4
    )
    assert mask.shape == (1, 2, 4, 4)
    expected = make_causal_mask(4)[0, 0]
    assert jnp.array_equal(mask[0, 0], expected)


def test_mask_mod_to_score_mod_round_trip():
    """mask_mod_to_score_mod must mirror passing the mask explicitly."""

    def causal(b, h, q, kv):
        return kv <= q

    q, k, v = _qkv(seq=8)
    out_mask = standard_attention(q, k, v, mask=make_causal_mask(8))
    out_mod = standard_attention(
        q, k, v, score_mod=mask_mod_to_score_mod(causal)
    )
    assert jnp.allclose(out_mask, out_mod, atol=1e-5)


def test_strategy_pattern_custom_attention_fn():
    """MultiHeadAttention should accept a custom AttentionFn and use it."""
    calls = {"n": 0}

    def my_kernel(query, key, value, **kwargs):
        calls["n"] += 1
        return standard_attention(query, key, value, **kwargs)

    rngs = nnx.Rngs(0)
    attn = MultiHeadAttention(
        rngs,
        num_heads=4,
        in_features=64,
        attention_fn=my_kernel,
    )
    x = jax.random.normal(jax.random.key(1), (2, 6, 64))
    y = attn(x, deterministic=True)
    assert y.shape == x.shape
    assert calls["n"] == 1


def test_strategy_pattern_forwards_score_mod_and_mask():
    """Custom kernel must receive the layer-configured score_mod and mask."""
    seen = {}

    def spy_kernel(query, key, value, **kwargs):
        seen["score_mod"] = kwargs.get("score_mod")
        seen["mask_shape"] = (
            None if kwargs.get("mask") is None else kwargs["mask"].shape
        )
        return standard_attention(query, key, value, **kwargs)

    rngs = nnx.Rngs(0)
    attn = MultiHeadAttention(
        rngs,
        num_heads=4,
        in_features=64,
        attention_fn=spy_kernel,
        score_mod=causal_mod(),
        attention_window=4,
    )
    x = jax.random.normal(jax.random.key(2), (2, 8, 64))
    attn(x, deterministic=True)
    # window + causal mod must compose to a non-None score_mod, and the
    # sliding-window boolean mask must also be applied.
    assert seen["score_mod"] is not None
    assert seen["mask_shape"] == (1, 1, 8, 8)


def test_strategy_pattern_per_call_score_mod_composes_on_top():
    """A per-call score_mod must stack on the constructor-supplied one."""
    seen = {}

    def spy_kernel(query, key, value, **kwargs):
        seen["score_mod"] = kwargs.get("score_mod")
        return standard_attention(query, key, value, **kwargs)

    rngs = nnx.Rngs(0)
    attn = MultiHeadAttention(
        rngs,
        num_heads=4,
        in_features=64,
        attention_fn=spy_kernel,
        score_mod=causal_mod(),
    )
    x = jax.random.normal(jax.random.key(3), (1, 6, 64))
    attn(x, score_mod=alibi_mod(num_heads=4), deterministic=True)
    assert seen["score_mod"] is not None


def test_lite_attention_rejects_score_mod():
    """LITE backend must raise rather than silently ignore score_mod."""
    rngs = nnx.Rngs(0)
    attn = MultiHeadAttention(
        rngs,
        num_heads=4,
        in_features=64,
        attention_type=AttentionType.LITE,
    )
    x = jax.random.normal(jax.random.key(4), (1, 5, 64))
    with pytest.raises(NotImplementedError):
        attn(x, score_mod=causal_mod(), deterministic=True)


def test_sliding_window_in_multihead_attention_matches_explicit_mask():
    """attention_window must match passing the window mask manually."""
    rngs = nnx.Rngs(0)
    x = jax.random.normal(jax.random.key(5), (1, 12, 64))

    attn_window = MultiHeadAttention(
        rngs, num_heads=4, in_features=64, attention_window=4,
    )
    rngs_b = nnx.Rngs(0)
    attn_manual = MultiHeadAttention(
        rngs_b, num_heads=4, in_features=64,
    )
    mask = make_sliding_window_mask(12, window_size=4, causal=True)
    y_a = attn_window(x, deterministic=True)
    y_b = attn_manual(x, mask=mask, deterministic=True)
    assert jnp.allclose(y_a, y_b, atol=1e-4)


def test_alibi_encoder_runs():
    """TransformerEncoder must run with pos_emb_type='alibi'."""
    config = TransformerConfig(
        vocab_size=300,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        pos_emb_type="alibi",
        max_len=64,
    )
    enc = TransformerEncoder(nnx.Rngs(0), config)
    ids = jnp.ones((2, 8), dtype=jnp.int32)
    out = enc(ids, deterministic=True)
    assert out.shape == (2, 8, 64)


def test_sliding_window_encoder_runs():
    """TransformerEncoder with attention_window= runs end-to-end."""
    config = TransformerConfig(
        vocab_size=300,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        attention_window=4,
        max_len=64,
    )
    enc = TransformerEncoder(nnx.Rngs(0), config)
    ids = jnp.ones((2, 12), dtype=jnp.int32)
    out = enc(ids, deterministic=True)
    assert out.shape == (2, 12, 64)


def test_score_mod_works_under_jit():
    """Score-mods must compose with jax.jit."""
    q, k, v = _qkv(seq=8)
    mod = alibi_mod(num_heads=4)
    f = jax.jit(lambda q, k, v: standard_attention(q, k, v, score_mod=mod))
    out = f(q, k, v)
    assert out.shape == q.shape


def test_flash_falls_back_with_score_mod_on_cpu():
    """flash_attention with score_mod set must fall back gracefully on CPU."""
    q, k, v = _qkv(seq=64)
    out = flash_attention(q, k, v, score_mod=causal_mod(), block_size=32)
    out_mem = memory_efficient_attention(
        q, k, v, score_mod=causal_mod(), block_size=32
    )
    assert jnp.allclose(out, out_mem, atol=1e-5)


def main():
    print("=" * 60)
    print("Running kernel tests")
    print("=" * 60)
    test_standard_matches_memory_efficient_small()
    test_memory_efficient_online_softmax_correct_for_long_seqs()
    test_zero_score_mod_is_identity()
    test_causal_mod_matches_causal_mask()
    test_sliding_window_mod_matches_mask()
    test_sliding_window_mask_shape_and_pattern()
    test_alibi_slopes_shape_and_sign()
    test_alibi_slopes_non_power_of_two()
    test_alibi_mod_adds_negative_distance_bias()
    test_prefix_lm_mod_pattern()
    test_document_mask_isolates_documents()
    test_document_mask_mod_matches_explicit_mask()
    test_compose_score_mods_order()
    test_compose_score_mods_none_passthrough()
    test_mask_mod_to_boolean_mask_round_trip()
    test_mask_mod_to_score_mod_round_trip()
    test_strategy_pattern_custom_attention_fn()
    test_strategy_pattern_forwards_score_mod_and_mask()
    test_strategy_pattern_per_call_score_mod_composes_on_top()
    test_sliding_window_in_multihead_attention_matches_explicit_mask()
    test_alibi_encoder_runs()
    test_sliding_window_encoder_runs()
    test_score_mod_works_under_jit()
    test_flash_falls_back_with_score_mod_on_cpu()
    print("\n" + "=" * 60)
    print("✓ All kernel tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
