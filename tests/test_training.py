# SPDX-License-Identifier: Apache-2.0

"""Training loop demonstration."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from attnax import TransformerEncoder, TransformerConfig, make_padding_mask


class TransformerClassifier(nnx.Module):
    """Transformer encoder with classification head."""

    def __init__(self, encoder: TransformerEncoder, num_classes: int, rngs: nnx.Rngs):
        self.encoder = encoder
        self.classifier = nnx.Linear(encoder.config.d_model, num_classes, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        padding_mask: jnp.ndarray | None = None,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
          input_ids: Token IDs.
          padding_mask: Optional padding mask.
          deterministic: If True, disables dropout.

        Returns:
          Classification logits of shape (batch, num_classes).
        """
        x = self.encoder(
            input_ids, padding_mask=padding_mask, deterministic=deterministic
        )
        return self.classifier(x[:, 0, :])


def create_model_and_optimizer(
    config: TransformerConfig, num_classes: int
) -> tuple[TransformerClassifier, nnx.Optimizer]:
    """Creates model and optimizer."""
    rngs = nnx.Rngs(42)
    encoder = TransformerEncoder(rngs, config)
    model = TransformerClassifier(encoder, num_classes=num_classes, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    return model, optimizer


def train_step(
    model: TransformerClassifier,
    optimizer: nnx.Optimizer,
    input_ids: jnp.ndarray,
    labels: jnp.ndarray,
    padding_mask: jnp.ndarray,
) -> float:
    """Single training step."""

    def loss_fn(model: TransformerClassifier) -> float:
        logits = model(input_ids, padding_mask=padding_mask, deterministic=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)
    return float(loss)


def eval_step(
    model: TransformerClassifier,
    input_ids: jnp.ndarray,
    labels: jnp.ndarray,
    padding_mask: jnp.ndarray,
) -> tuple[float, float]:
    """Single evaluation step."""
    logits = model(input_ids, padding_mask=padding_mask, deterministic=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return float(loss), float(accuracy)


def test_training_loop():
    """Test that training loop works correctly."""
    print("=" * 70)
    print("Training loop demonstration")
    print("=" * 70)

    batch_size, seq_len = 8, 32
    vocab_size, num_classes = 1000, 10
    num_steps = 20

    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        dropout_rate=0.1,
    )

    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary: {vocab_size}")
    print(f"  Classes: {num_classes}")
    print(f"  Steps: {num_steps}")

    model, optimizer = create_model_and_optimizer(config, num_classes)

    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    input_ids = jax.random.randint(key1, (batch_size, seq_len), 0, vocab_size)
    labels = jax.random.randint(key2, (batch_size,), 0, num_classes)
    padding_mask = make_padding_mask(input_ids, pad_token_id=0)

    print(f"\n{'Step':<8} {'Train Loss':<12} {'Eval Loss':<12} {'Accuracy':<10}")
    print("-" * 50)

    for step in range(num_steps):
        train_loss = train_step(model, optimizer, input_ids, labels, padding_mask)

        if step % 5 == 0 or step == num_steps - 1:
            eval_loss, accuracy = eval_step(model, input_ids, labels, padding_mask)
            print(f"{step:<8} {train_loss:<12.6f} {eval_loss:<12.6f} {accuracy:<10.4f}")

    # Verify training worked
    final_loss, final_accuracy = eval_step(model, input_ids, labels, padding_mask)
    assert final_loss < 10.0, f"Loss too high: {final_loss}"
    assert final_accuracy >= 0.0, f"Invalid accuracy: {final_accuracy}"
    assert optimizer.step.value == num_steps, (
        f"Wrong step count: {optimizer.step.value}"
    )

    print("\n" + "=" * 70)
    print("✓ Training completed")
    print(f"  Final step: {optimizer.step.value}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Final accuracy: {final_accuracy:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    test_training_loop()
