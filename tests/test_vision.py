# SPDX-License-Identifier: Apache-2.0

"""Component tests for the Vision Transformer (ViT) stack."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import pytest

from attnax import (
    AttentionType,
    LearnedPositionalEmbedding,
    PatchEmbedding,
    VisionTransformer,
    VisionTransformerConfig,
)


def test_patch_embedding_shapes():
    """PatchEmbedding converts (B, H, W, C) into (B, num_patches, d_model)."""
    print("\n=== Testing PatchEmbedding ===")
    rngs = nnx.Rngs(0)
    patch = PatchEmbedding(
        rngs,
        image_size=32,
        patch_size=8,
        num_channels=3,
        d_model=64,
    )
    imgs = jnp.zeros((2, 32, 32, 3))
    tokens = patch(imgs)
    assert tokens.shape == (2, 16, 64)  # 16 = (32 // 8) ** 2
    assert patch.num_patches == 16
    assert patch.grid_size == (4, 4)
    print(f"✓ PatchEmbedding: {tokens.shape}")


def test_patch_embedding_rectangular():
    """PatchEmbedding accepts rectangular images / patches."""
    rngs = nnx.Rngs(1)
    patch = PatchEmbedding(
        rngs,
        image_size=(32, 16),
        patch_size=(8, 4),
        num_channels=1,
        d_model=32,
    )
    imgs = jnp.zeros((2, 32, 16, 1))
    tokens = patch(imgs)
    assert tokens.shape == (2, 16, 32)  # (32/8) * (16/4) = 4 * 4 = 16
    print(f"✓ PatchEmbedding (rectangular): {tokens.shape}")


def test_patch_embedding_invalid():
    """PatchEmbedding rejects non-divisible image / patch sizes."""
    rngs = nnx.Rngs(2)
    with pytest.raises(ValueError):
        PatchEmbedding(
            rngs, image_size=30, patch_size=8, num_channels=3, d_model=32
        )


def test_learned_positional_embedding():
    """LearnedPositionalEmbedding is additive and preserves shape."""
    print("\n=== Testing LearnedPositionalEmbedding ===")
    rngs = nnx.Rngs(3)
    pos = LearnedPositionalEmbedding(
        rngs, num_positions=17, d_model=64
    )
    x = jnp.zeros((2, 17, 64))
    y = pos(x)
    assert y.shape == x.shape
    # Non-trivial: positional embedding should not be all-zeros at init.
    assert not jnp.allclose(y, 0.0)
    print(f"✓ LearnedPositionalEmbedding: {y.shape}")


def test_learned_positional_overflow():
    """Sequence longer than the table raises ValueError."""
    rngs = nnx.Rngs(4)
    pos = LearnedPositionalEmbedding(
        rngs, num_positions=10, d_model=32
    )
    x = jnp.zeros((1, 11, 32))
    with pytest.raises(ValueError):
        pos(x)


def test_vit_classification_head():
    """ViT with classification head returns logits."""
    print("\n=== Testing VisionTransformer (classification) ===")
    config = VisionTransformerConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        num_classes=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        dropout_rate=0.1,
    )
    model = VisionTransformer(nnx.Rngs(5), config)
    imgs = jax.random.normal(jax.random.key(0), (2, 32, 32, 3))
    logits_inf = model(imgs, deterministic=True)
    assert logits_inf.shape == (2, config.num_classes)
    logits_train = model(imgs, deterministic=False)
    assert logits_train.shape == (2, config.num_classes)
    print(f"✓ ViT logits: {logits_inf.shape}")


def test_vit_backbone_mode():
    """ViT without a head returns the full token sequence."""
    print("\n=== Testing VisionTransformer (backbone) ===")
    config = VisionTransformerConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        num_classes=None,
        d_model=64,
        num_heads=4,
        num_layers=1,
        d_ff=128,
    )
    model = VisionTransformer(nnx.Rngs(6), config)
    imgs = jnp.zeros((2, 32, 32, 3))
    tokens = model(imgs, deterministic=True)
    # 16 patches + 1 CLS token
    assert tokens.shape == (2, 17, 64)
    print(f"✓ ViT backbone tokens: {tokens.shape}")


def test_vit_mean_pool_no_cls():
    """ViT with mean pooling and no CLS token."""
    print("\n=== Testing VisionTransformer (mean pool, no CLS) ===")
    config = VisionTransformerConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        num_classes=4,
        d_model=64,
        num_heads=4,
        num_layers=1,
        d_ff=128,
        use_cls_token=False,
        pool="mean",
    )
    model = VisionTransformer(nnx.Rngs(7), config)
    imgs = jnp.zeros((2, 32, 32, 3))
    logits = model(imgs, deterministic=True)
    assert logits.shape == (2, 4)
    print(f"✓ ViT mean-pool logits: {logits.shape}")


def test_vit_modern_options():
    """ViT with RMSNorm + SwiGLU + GQA + memory-efficient attention."""
    print("\n=== Testing VisionTransformer (modern options) ===")
    config = VisionTransformerConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        num_classes=5,
        d_model=64,
        num_heads=8,
        num_layers=2,
        d_ff=128,
        norm_type="rms",
        ff_activation="swiglu",
        num_kv_heads=2,
        attention_type=AttentionType.MEMORY_EFFICIENT,
    )
    model = VisionTransformer(nnx.Rngs(8), config)
    imgs = jnp.zeros((2, 32, 32, 3))
    logits = model(imgs, deterministic=True)
    assert logits.shape == (2, 5)
    print(f"✓ ViT (RMS+SwiGLU+GQA+mem-eff): {logits.shape}")


def test_vit_invalid_config():
    """VisionTransformerConfig validates pool + cls combination and divisibility."""
    with pytest.raises(ValueError):
        VisionTransformerConfig(use_cls_token=False, pool="cls")
    with pytest.raises(ValueError):
        VisionTransformerConfig(image_size=224, patch_size=17)


def test_vit_training_step():
    """A single optimizer step on the ViT actually decreases the loss."""
    print("\n=== Testing VisionTransformer (training step) ===")
    config = VisionTransformerConfig(
        image_size=16,
        patch_size=4,
        num_channels=3,
        num_classes=5,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        dropout_rate=0.0,
    )
    model = VisionTransformer(nnx.Rngs(9), config)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    imgs = jax.random.normal(jax.random.key(0), (4, 16, 16, 3))
    labels = jnp.array([0, 1, 2, 3])

    def loss_fn(model):
        logits = model(imgs, deterministic=True)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()

    loss_before = loss_fn(model)
    _, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)
    loss_after = loss_fn(model)
    assert float(loss_after) < float(loss_before)
    print(
        f"✓ ViT training step: loss {float(loss_before):.4f} → "
        f"{float(loss_after):.4f}"
    )


def main():
    print("=" * 60)
    print("Running Vision Transformer tests")
    print("=" * 60)

    test_patch_embedding_shapes()
    test_patch_embedding_rectangular()
    test_patch_embedding_invalid()
    test_learned_positional_embedding()
    test_learned_positional_overflow()
    test_vit_classification_head()
    test_vit_backbone_mode()
    test_vit_mean_pool_no_cls()
    test_vit_modern_options()
    test_vit_invalid_config()
    test_vit_training_step()

    print("\n" + "=" * 60)
    print("✓ All Vision Transformer tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
