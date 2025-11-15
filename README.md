# Attnax

Composable attention and transformer components for JAX.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax) [![Flax NNX](https://img.shields.io/badge/Flax-NNX-green.svg)](https://flax.readthedocs.io/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Installation](#installation) | [Quick Start](#quick-start) | [API Reference](#api-reference) | [Examples](examples/)

## What is Attnax?

Attnax is a library of transformer primitives built on JAX and Flax NNX. It provides modular, composable components for building transformer architectures without rewriting standard building blocks from scratch.

The library includes:
- Multi-head attention (self and cross)
- Position-wise feed-forward networks
- Token and positional embeddings (sinusoidal)
- Encoder and decoder blocks
- Masking utilities (padding, causal)

All components are implemented using Flax NNX with full type annotations and can be composed to build custom transformer architectures. JAX transformations (`jit`, `vmap`, `grad`) work naturally with all modules.

```python
import jax.numpy as jnp
import flax.nnx as nnx
from attnax import TransformerConfig, TransformerEncoder

config = TransformerConfig(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    num_layers=6,
)

model = TransformerEncoder(nnx.Rngs(42), config)
output = model(jnp.ones((2, 10), dtype=jnp.int32), deterministic=True)
print(output.shape)  # (2, 10, 512)
```

## Installation

```bash
pip install attnax
```

Or install from source:

```bash
git clone https://github.com/glibtkachenko/attnax.git
cd attnax
pip install -e .
```

Requires Python 3.9+, JAX 0.4.0+, Flax 0.8.0+, and Optax 0.1.0+.

## Quick Start

### Basic encoder

```python
import jax.numpy as jnp
import flax.nnx as nnx
from attnax import TransformerConfig, TransformerEncoder

config = TransformerConfig(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    dropout_rate=0.1,
    max_seq_len=512,
)

rngs = nnx.Rngs(42)
model = TransformerEncoder(rngs, config)

input_ids = jnp.ones((2, 10), dtype=jnp.int32)
output = model(input_ids, deterministic=True)  # (2, 10, 512)
```

### With padding masks

```python
from attnax import make_padding_mask

input_ids = jnp.array([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
padding_mask = make_padding_mask(input_ids, pad_token_id=0)

output = model(input_ids, padding_mask=padding_mask, deterministic=True)
```

### Training

```python
import optax

optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)

def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch['input_ids'], deterministic=False)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['labels']
        ).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)
    return loss

for batch in dataloader:
    loss = train_step(model, optimizer, batch)
```

## API Reference

### Configuration

#### `TransformerConfig`

Dataclass containing all transformer hyperparameters.

```python
config = TransformerConfig(
    vocab_size=32000,        # Size of vocabulary
    d_model=512,             # Model dimension
    num_heads=8,             # Number of attention heads
    num_layers=6,            # Number of encoder/decoder layers
    d_ff=2048,               # Feed-forward dimension
    dropout_rate=0.1,        # Dropout probability
    max_seq_len=512,         # Maximum sequence length
    activation='gelu',       # Activation function ('gelu', 'relu', 'swish')
    use_bias=True,           # Whether to use bias in linear layers
    layer_norm_eps=1e-6,     # Layer normalization epsilon
    pad_token_id=0,          # Padding token ID
)
```

### Core Modules

#### `TransformerEncoder`

Complete transformer encoder with token embeddings, positional encoding, and stacked encoder blocks.

```python
encoder = TransformerEncoder(rngs, config)

output = encoder(
    input_ids,                    # Shape: (batch, seq_len)
    padding_mask=None,            # Shape: (batch, 1, 1, seq_len)
    deterministic=True,           # Disable dropout for inference
)
# Returns: (batch, seq_len, d_model)
```

#### `EncoderBlock`

Single transformer encoder block with self-attention and feed-forward network.

```python
block = EncoderBlock(rngs, config)

output = block(
    x,                            # Shape: (batch, seq_len, d_model)
    padding_mask=None,            # Shape: (batch, 1, 1, seq_len)
    deterministic=True,
)
# Returns: (batch, seq_len, d_model)
```

#### `MultiHeadAttentionLayer`

Multi-head attention with support for both self-attention and cross-attention.

```python
attention = MultiHeadAttentionLayer(rngs, config)

# Self-attention
output = attention(x, deterministic=True)

# Cross-attention
output = attention(x, context=encoder_output, mask=mask, deterministic=True)
# Returns: (batch, seq_len, d_model)
```

#### `FeedForward`

Position-wise feed-forward network with configurable activation.

```python
ffn = FeedForward(rngs, config)

output = ffn(x, deterministic=True)
# Returns: (batch, seq_len, d_model)
```

#### `TokenEmbedding`

Token embedding layer.

```python
embedding = TokenEmbedding(config.vocab_size, config.d_model, rngs)

embedded = embedding(input_ids)
# Returns: (batch, seq_len, d_model)
```

#### `PositionalEncoding`

Sinusoidal positional encoding.

```python
pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)

encoded = pos_encoding(x)
# Returns: (batch, seq_len, d_model)
```

### Masking Utilities

#### `make_padding_mask`

Creates padding mask from input token IDs.

```python
mask = make_padding_mask(input_ids, pad_token_id=0)
# Returns: (batch, 1, 1, seq_len) boolean mask
```

#### `make_causal_mask`

Creates causal mask for autoregressive attention.

```python
mask = make_causal_mask(seq_len)
# Returns: (1, 1, seq_len, seq_len) boolean mask
```

#### `combine_masks`

Combines multiple masks via logical AND.

```python
combined = combine_masks(padding_mask, causal_mask)
# Returns: Combined boolean mask
```

## Components

### Core modules

- `TransformerEncoder` - Complete encoder with embeddings and stacked blocks
- `EncoderBlock` - Single encoder layer with self-attention and FFN
- `DecoderBlock` - Single decoder layer with self-attention, cross-attention, and FFN
- `MultiHeadAttentionLayer` - Multi-head attention (self or cross)
- `FeedForward` - Position-wise feed-forward network
- `TokenEmbedding` - Token embedding layer
- `PositionalEncoding` - Sinusoidal positional encoding

### Masking utilities

- `make_padding_mask` - Creates padding masks from token IDs
- `make_causal_mask` - Creates causal masks for autoregressive decoding
- `combine_masks` - Combines multiple masks via logical AND

See the [API Reference](#api-reference) section for detailed documentation.

## Advanced usage

### Custom architectures

Build custom models by composing components:

```python
import flax.nnx as nnx
from jaxtransformer import EncoderBlock, TokenEmbedding, PositionalEncoding

class CustomTransformer(nnx.Module):
    def __init__(self, rngs, config):
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model, rngs)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)

        # Custom layer configuration
        self.blocks = nnx.List([
            EncoderBlock(rngs, config) for _ in range(config.num_layers)
        ])

        self.output_projection = nnx.Linear(config.d_model, config.vocab_size, rngs=rngs)

    def __call__(self, input_ids, deterministic=True):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x, deterministic=deterministic)

        return self.output_projection(x)
```

### Model serialization

Save and load model checkpoints:

```python
import orbax.checkpoint as ocp

checkpointer = ocp.StandardCheckpointer()
state = nnx.state(model)
checkpointer.save(f'checkpoints/step_{step}', state)

restored_state = checkpointer.restore('checkpoints/step_1000')
nnx.update(model, restored_state)
```

### Multi-device training

Use JAX's `pmap` for data parallelism:

```python
@jax.pmap
def parallel_train_step(model, batch):
    def loss_fn(model):
        logits = model(batch['input_ids'], deterministic=False)
        return compute_loss(logits, batch['labels'])

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    grads = jax.lax.pmean(grads, axis_name='batch')
    optimizer.update(model=model, grads=grads)
    return loss
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Individual test modules:

```bash
python -m tests.test_components
python -m tests.test_training
```

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
