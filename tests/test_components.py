# SPDX-License-Identifier: Apache-2.0

"""Component tests for transformer modules."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from attnax import (
    MultiHeadAttention,
    TransformerDecoder,
    TransformerEncoder,
    TransformerConfig,
    TokenEmbedding,
    PositionalEncoding,
    FeedForward,
    EncoderBlock,
    DecoderBlock,
    make_padding_mask,
    make_causal_mask,
    combine_masks,
    init_kv_layer_cache,
    init_decoder_kv_caches_from_config,
)


def test_embeddings():
    """Test token and positional embeddings."""
    print("\n=== Testing Embeddings ===")
    rngs = nnx.Rngs(0)
    vocab_size, d_model = 1000, 128
    batch_size, seq_len = 4, 10

    token_embed = TokenEmbedding(rngs, vocab_size, d_model)
    token_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    embeddings = token_embed(token_ids)
    assert embeddings.shape == (batch_size, seq_len, d_model)
    print(f"✓ Token embeddings: {embeddings.shape}")

    pos_enc = PositionalEncoding(512, d_model)
    output = pos_enc(embeddings)
    assert output.shape == embeddings.shape
    print(f"✓ Positional encoding: {output.shape}")


def test_feedforward():
    """Test feed-forward network."""
    print("\n=== Testing FeedForward ===")
    rngs = nnx.Rngs(42)
    d_model, d_ff = 128, 512
    batch_size, seq_len = 4, 10

    ffn = FeedForward(rngs, d_model, d_ff, dropout_rate=0.1)
    x = jnp.ones((batch_size, seq_len, d_model))

    output_inf = ffn(x, deterministic=True)
    assert output_inf.shape == x.shape
    print(f"✓ FFN (inference): {output_inf.shape}")

    output_train = ffn(x, deterministic=False)
    assert output_train.shape == x.shape
    print(f"✓ FFN (training): {output_train.shape}")


def test_attention():
    """Test multi-head attention."""
    print("\n=== Testing MultiHeadAttention ===")
    rngs = nnx.Rngs(123)
    num_heads, d_model = 4, 128
    batch_size, seq_len = 2, 10

    attn = MultiHeadAttention(
        rngs, num_heads=num_heads, in_features=d_model, dropout_rate=0.1
    )
    x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, d_model))

    output_self = attn(x, deterministic=True)
    assert output_self.shape == x.shape
    print(f"✓ Self-attention: {output_self.shape}")

    context = jax.random.normal(jax.random.key(1), (batch_size, 15, d_model))
    output_cross = attn(x, context=context, deterministic=True)
    assert output_cross.shape == x.shape
    print(f"✓ Cross-attention: {output_cross.shape}")


def test_blocks():
    """Test encoder and decoder blocks."""
    print("\n=== Testing Blocks ===")
    rngs = nnx.Rngs(456)
    d_model, num_heads, d_ff = 128, 4, 512
    batch_size, seq_len = 2, 10

    encoder_block = EncoderBlock(rngs, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, d_model))
    output = encoder_block(x, deterministic=True)
    assert output.shape == x.shape
    print(f"✓ EncoderBlock: {output.shape}")

    decoder_block = DecoderBlock(rngs, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    encoder_output = jax.random.normal(jax.random.key(1), (batch_size, 15, d_model))
    output = decoder_block(x, encoder_output=encoder_output, deterministic=True)
    assert output.shape == x.shape
    print(f"✓ DecoderBlock: {output.shape}")


def test_encoder():
    """Test transformer encoder."""
    print("\n=== Testing TransformerEncoder ===")
    config = TransformerConfig(
        vocab_size=5000,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=512,
        dropout_rate=0.1,
    )
    rngs = nnx.Rngs(999)
    batch_size, seq_len = 2, 20

    encoder = TransformerEncoder(rngs, config)
    input_ids = jax.random.randint(
        jax.random.key(0), (batch_size, seq_len), 0, config.vocab_size
    )
    padding_mask = make_padding_mask(input_ids, pad_token_id=config.pad_token_id)

    output_inf = encoder(input_ids, padding_mask=padding_mask, deterministic=True)
    assert output_inf.shape == (batch_size, seq_len, config.d_model)
    print(f"✓ Encoder (inference): {output_inf.shape}")

    output_train = encoder(input_ids, padding_mask=padding_mask, deterministic=False)
    assert output_train.shape == (batch_size, seq_len, config.d_model)
    print(f"✓ Encoder (training): {output_train.shape}")


def test_masking():
    """Test masking utilities."""
    print("\n=== Testing Masking ===")
    batch_size, seq_len = 2, 10

    input_ids = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]]
    )
    padding_mask = make_padding_mask(input_ids, pad_token_id=0)
    assert padding_mask.shape == (batch_size, 1, 1, seq_len)
    print(f"✓ Padding mask: {padding_mask.shape}")

    causal_mask = make_causal_mask(seq_len)
    assert causal_mask.shape == (1, 1, seq_len, seq_len)
    print(f"✓ Causal mask: {causal_mask.shape}")

    combined = combine_masks(padding_mask, causal_mask)
    assert combined.shape == (batch_size, 1, seq_len, seq_len)
    print(f"✓ Combined mask: {combined.shape}")


def test_rope_encoder():
    """Encoder with RoPE positional type."""
    print("\n=== Testing RoPE encoder ===")
    config = TransformerConfig(
        vocab_size=500,
        d_model=128,
        num_heads=4,
        num_layers=1,
        d_ff=256,
        pos_emb_type="rope",
        max_len=64,
    )
    rngs = nnx.Rngs(11)
    enc = TransformerEncoder(rngs, config)
    batch, seq = 2, 12
    ids = jax.random.randint(jax.random.key(0), (batch, seq), 0, config.vocab_size)
    out = enc(ids, deterministic=True)
    assert out.shape == (batch, seq, config.d_model)
    print(f"✓ RoPE encoder: {out.shape}")


def test_gqa_attention():
    """Grouped-query attention with fewer KV heads."""
    print("\n=== Testing GQA ===")
    rngs = nnx.Rngs(7)
    d_model, num_heads = 128, 8
    attn = MultiHeadAttention(
        rngs,
        num_heads=num_heads,
        in_features=d_model,
        num_kv_heads=2,
        dropout_rate=0.0,
    )
    x = jax.random.normal(jax.random.key(0), (2, 16, d_model))
    y = attn(x, deterministic=True)
    assert y.shape == x.shape
    print(f"✓ GQA self-attention: {y.shape}")


def test_rms_encoder():
    """Encoder with RMSNorm."""
    print("\n=== Testing RMSNorm encoder ===")
    config = TransformerConfig(
        vocab_size=300,
        d_model=64,
        num_heads=4,
        num_layers=1,
        d_ff=128,
        norm_type="rms",
    )
    rngs = nnx.Rngs(13)
    enc = TransformerEncoder(rngs, config)
    ids = jnp.ones((2, 8), dtype=jnp.int32)
    out = enc(ids, deterministic=True)
    assert out.shape == (2, 8, 64)
    print(f"✓ RMSNorm encoder: {out.shape}")


def test_swiglu_ffn():
    """SwiGLU feed-forward."""
    print("\n=== Testing SwiGLU FFN ===")
    rngs = nnx.Rngs(17)
    ffn = FeedForward(rngs, d_model=64, d_ff=128, ff_activation="swiglu")
    x = jnp.ones((2, 5, 64))
    y = ffn(x, deterministic=True)
    assert y.shape == x.shape
    print(f"✓ SwiGLU: {y.shape}")


def test_kv_cache_self_attention():
    """Prefill + one-token decode with KV cache."""
    print("\n=== Testing KV cache (MHA) ===")
    rngs = nnx.Rngs(101)
    d_model, num_heads = 64, 4
    attn = MultiHeadAttention(
        rngs,
        num_heads=num_heads,
        in_features=d_model,
        dropout_rate=0.0,
    )
    batch, max_len = 2, 32
    head_dim = d_model // num_heads
    cache = init_kv_layer_cache(
        batch, num_heads, head_dim, max_len, jnp.float32
    )
    x_prefill = jax.random.normal(jax.random.key(1), (batch, 4, d_model))
    mask_prefill = jnp.ones((batch, 1, 4, 4), dtype=bool)
    o1, c1 = attn(
        x_prefill, mask=mask_prefill, kv_cache=cache, deterministic=True
    )
    assert o1.shape == (batch, 4, d_model)
    assert int(c1.length) == 4
    x_step = jax.random.normal(jax.random.key(2), (batch, 1, d_model))
    mask_step = jnp.ones((batch, 1, 1, 5), dtype=bool)
    o2, c2 = attn(
        x_step, mask=mask_step, kv_cache=c1, deterministic=True
    )
    assert o2.shape == (batch, 1, d_model)
    assert int(c2.length) == 5
    print(f"✓ KV cache MHA: prefill {o1.shape}, decode {o2.shape}")


def test_kv_cache_gqa():
    """KV cache with fewer KV heads."""
    print("\n=== Testing KV cache (GQA) ===")
    rngs = nnx.Rngs(103)
    d_model, num_heads, num_kv = 64, 8, 2
    attn = MultiHeadAttention(
        rngs,
        num_heads=num_heads,
        in_features=d_model,
        num_kv_heads=num_kv,
        dropout_rate=0.0,
    )
    batch = 1
    head_dim = d_model // num_heads
    cache = init_kv_layer_cache(batch, num_kv, head_dim, 64, jnp.float32)
    x = jax.random.normal(jax.random.key(0), (batch, 3, d_model))
    m = jnp.ones((batch, 1, 3, 3), dtype=bool)
    o, c = attn(x, mask=m, kv_cache=cache, deterministic=True)
    assert o.shape == x.shape
    assert int(c.length) == 3
    print(f"✓ KV cache GQA: {o.shape}")


def test_encoder_layer_kv_caches():
    """TransformerEncoder with per-layer KV caches."""
    print("\n=== Testing encoder + layer KV caches ===")
    config = TransformerConfig(
        vocab_size=200,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=64,
        max_len=64,
    )
    rngs = nnx.Rngs(107)
    enc = TransformerEncoder(rngs, config)
    caches = init_decoder_kv_caches_from_config(
        config, batch_size=1, max_len=64
    )
    ids = jnp.ones((1, 3), dtype=jnp.int32)
    out, new_caches = enc(
        ids, layer_kv_caches=caches, deterministic=True
    )
    assert out.shape == (1, 3, 32)
    assert len(new_caches) == 2
    assert int(new_caches[0].length) == 3
    print(f"✓ Encoder KV caches: {out.shape}")


def test_decoder_training():
    """TransformerDecoder forward pass applies causal mask internally."""
    print("\n=== Testing TransformerDecoder (training) ===")
    config = TransformerConfig(
        vocab_size=500,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        pos_emb_type="rope",
        max_len=64,
    )
    rngs = nnx.Rngs(201)
    dec = TransformerDecoder(rngs, config)
    batch, seq_len = 2, 12
    ids = jax.random.randint(
        jax.random.key(0), (batch, seq_len), 0, config.vocab_size
    )
    out = dec(ids, deterministic=True)
    assert out.shape == (batch, seq_len, config.d_model)
    print(f"✓ Decoder (training): {out.shape}")


def test_decoder_causal_isolation():
    """Future tokens must not affect earlier outputs (causality check)."""
    print("\n=== Testing TransformerDecoder causality ===")
    config = TransformerConfig(
        vocab_size=200,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        dropout_rate=0.0,
        pos_emb_type="rope",
        max_len=32,
    )
    rngs = nnx.Rngs(202)
    dec = TransformerDecoder(rngs, config)
    seq_len = 8
    ids_a = jax.random.randint(
        jax.random.key(0), (1, seq_len), 0, config.vocab_size
    )
    # Replace the last 3 tokens with different values; outputs at positions
    # 0..seq_len-4 must remain identical because of causal masking.
    ids_b = ids_a.at[:, -3:].set(
        (ids_a[:, -3:] + 1) % config.vocab_size
    )
    out_a = dec(ids_a, deterministic=True)
    out_b = dec(ids_b, deterministic=True)
    assert jnp.allclose(out_a[:, :-3], out_b[:, :-3], atol=1e-5), (
        "Decoder is not causal: changing future tokens affected past outputs."
    )
    print(f"✓ Decoder causality preserved at positions [:{seq_len - 3}]")


def test_decoder_kv_cache_decode():
    """Prefill + step-by-step decode produces same output as a full pass."""
    print("\n=== Testing TransformerDecoder + KV cache ===")
    config = TransformerConfig(
        vocab_size=200,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        dropout_rate=0.0,
        pos_emb_type="rope",
        max_len=32,
    )
    rngs = nnx.Rngs(203)
    dec = TransformerDecoder(rngs, config)
    prompt_len, step_count = 4, 3
    total_len = prompt_len + step_count
    ids = jax.random.randint(
        jax.random.key(0), (1, total_len), 0, config.vocab_size
    )

    # Full pass (no cache) — reference output.
    full_out = dec(ids, deterministic=True)

    # Incremental pass: prefill then one token at a time.
    caches = init_decoder_kv_caches_from_config(
        config, batch_size=1, max_len=config.max_len
    )
    prefill_out, caches = dec(
        ids[:, :prompt_len], layer_kv_caches=caches, deterministic=True
    )
    assert prefill_out.shape == (1, prompt_len, config.d_model)
    assert int(caches[0].length) == prompt_len

    step_outs = [prefill_out]
    for i in range(step_count):
        step_ids = ids[:, prompt_len + i : prompt_len + i + 1]
        step_out, caches = dec(
            step_ids, layer_kv_caches=caches, deterministic=True
        )
        step_outs.append(step_out)
    incremental_out = jnp.concatenate(step_outs, axis=1)

    assert int(caches[0].length) == total_len
    assert jnp.allclose(full_out, incremental_out, atol=1e-4), (
        "Cached decode does not match full forward pass."
    )
    print(f"✓ Decoder KV cache matches full pass: {incremental_out.shape}")


def main():
    print("=" * 60)
    print("Running component tests")
    print("=" * 60)

    test_embeddings()
    test_feedforward()
    test_attention()
    test_blocks()
    test_encoder()
    test_masking()
    test_rope_encoder()
    test_gqa_attention()
    test_rms_encoder()
    test_swiglu_ffn()
    test_kv_cache_self_attention()
    test_kv_cache_gqa()
    test_encoder_layer_kv_caches()
    test_decoder_training()
    test_decoder_causal_isolation()
    test_decoder_kv_cache_decode()

    print("\n" + "=" * 60)
    print("✓ All component tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
