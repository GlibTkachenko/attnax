# Attnax: Attention for JAX

Attention kernels and transformer components for JAX.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax) [![Flax NNX](https://img.shields.io/badge/Flax-NNX-green.svg)](https://flax.readthedocs.io/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Installation](#installation) | [Quick start](#quick-start) | [Documentation](https://attnax.readthedocs.io) | [Examples](examples/) | [Citation](#citing-attnax)

## Overview

Attnax is built on [JAX](https://jax.readthedocs.io/) and
[Flax](https://flax.readthedocs.io/) and provides:

- Attention kernels as pure JAX functions sharing a single
  `AttentionFn` protocol: `standard_attention`,
  `memory_efficient_attention`, `flash_attention`, `linear_attention`,
  `ring_attention`, `pallas_flash_attention`, `paged_attention`,
  `lite_attention`.
- `ScoreMod` / `MaskMod` constructors for ALiBi, sliding window,
  prefix-LM, document masks, and arbitrary additive biases — composed
  with `compose_score_mods`.
- `MultiHeadAttention` with MHA, GQA, MQA, RoPE, sliding window, and
  optional contiguous or paged KV caching (`KVLayerCache`,
  `PagedKVCache`).
- `EncoderBlock`, `DecoderBlock`, `TransformerEncoder`,
  `TransformerDecoder`, `VisionTransformer`, `FeedForward`,
  `MixtureOfExperts`, `RMSNorm`, RoPE and the usual positional
  embeddings.

Documentation on Attnax can be found at [attnax.readthedocs.io](https://attnax.readthedocs.io).
## Installation

```bash
pip install attnax
```

From source:

```bash
git clone https://github.com/glibtkachenko/attnax.git
cd attnax
pip install -e .
```

Requires Python 3.10+, JAX 0.10.0+, and Flax 0.12.7+.

## Quick start

Attention kernels are pure JAX functions:

```python
import jax, jax.numpy as jnp
from attnax import standard_attention

q = jax.random.normal(jax.random.key(0), (1, 4, 64, 32))
k = jax.random.normal(jax.random.key(1), (1, 4, 64, 32))
v = jax.random.normal(jax.random.key(2), (1, 4, 64, 32))
out = standard_attention(q, k, v)
```

Biases compose as `ScoreMod`s:

```python
from attnax import alibi_mod, compose_score_mods, sliding_window_mod

mod = compose_score_mods(
    alibi_mod(num_heads=4),
    sliding_window_mod(window_size=128, causal=True),
)
out = standard_attention(q, k, v, score_mod=mod)
```

Any kernel matching `AttentionFn` plugs into `MultiHeadAttention`:

```python
import flax.nnx as nnx
from attnax import MultiHeadAttention, pallas_flash_attention

attn = MultiHeadAttention(
    nnx.Rngs(0),
    num_heads=8,
    in_features=512,
    attention_fn=pallas_flash_attention,
)
```

A full transformer stack:

```python
from attnax import TransformerConfig, TransformerEncoder

config = TransformerConfig(
    vocab_size=32000, d_model=512, num_heads=8, num_layers=6,
)
model = TransformerEncoder(nnx.Rngs(0), config)
y = model(jnp.ones((2, 16), dtype=jnp.int32), deterministic=True)
```

See the [getting-started notebook](docs/getting_started.ipynb) for a
walkthrough covering score-mods, custom kernels, KV caching, paged
caching, Mixture-of-Experts, the Vision Transformer, and training.

## Citing Attnax

```bibtex
@software{attnax2025github,
  author = {Glib Tkachenko},
  title = {{Attnax}: Attention Kernels and Transformer Components for {JAX}},
  url = {https://github.com/glibtkachenko/attnax},
  version = {0.2.0},
  year = {2025},
}
```
