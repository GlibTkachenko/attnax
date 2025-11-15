# SPDX-License-Identifier: Apache-2.0

"""Component tests for transformer modules."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from attnax import (
    MultiHeadAttentionLayer,
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

    attn = MultiHeadAttentionLayer(
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

    print("\n" + "=" * 60)
    print("✓ All component tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
