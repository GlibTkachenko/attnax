# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import attnax

pytestmark = pytest.mark.skipif(
  jax.default_backend() == "cpu",
  reason="GPU integration tests; current JAX backend is CPU.",
)

# Decoder-only LLM scale (≈ GPT-2 medium width, but fewer layers so the
# first-time XLA compilation cost stays bounded).
_BATCH = 4
_NUM_HEADS = 16
_SEQ = 2048
_HEAD_DIM = 64
_D_MODEL = _NUM_HEADS * _HEAD_DIM  # 1024
_D_FF = 4 * _D_MODEL
_VOCAB = 32_000
_NUM_LAYERS = 4


def _bench(label: str, fn: Callable, *args, repeats: int = 5):
  """JIT-compile, warm up, then time ``fn(*args)`` over ``repeats`` calls."""
  jit_fn = jax.jit(fn)
  out = jit_fn(*args)
  jax.block_until_ready(out)
  start = time.perf_counter()
  for _ in range(repeats):
    out = jit_fn(*args)
  jax.block_until_ready(out)
  per_iter_ms = (time.perf_counter() - start) * 1000 / repeats
  print(f"  {label}: {per_iter_ms:.2f} ms/iter ({repeats} iters)")
  return out


@pytest.fixture(scope="module")
def qkv():
  """Realistic Q/K/V tensors shared by every kernel test."""
  rng = jax.random.key(0)
  k0, k1, k2 = jax.random.split(rng, 3)
  shape = (_BATCH, _NUM_HEADS, _SEQ, _HEAD_DIM)
  q = jax.random.normal(k0, shape, dtype=jnp.float32)
  k = jax.random.normal(k1, shape, dtype=jnp.float32)
  v = jax.random.normal(k2, shape, dtype=jnp.float32)
  jax.block_until_ready((q, k, v))
  return q, k, v


@pytest.fixture(scope="module")
def reference(qkv):
  """``standard_attention`` reference output at LLM scale."""
  q, k, v = qkv
  ref = jax.jit(attnax.standard_attention)(q, k, v)
  jax.block_until_ready(ref)
  return ref


def test_device_is_gpu():
  """Report the active device and confirm it is not CPU."""
  devices = jax.devices()
  print(f"\n  JAX devices: {devices}")
  print(f"  Default backend: {jax.default_backend()}")
  print(
    f"  Shape under test: batch={_BATCH} heads={_NUM_HEADS} "
    f"seq={_SEQ} head_dim={_HEAD_DIM}"
  )
  assert jax.default_backend() != "cpu"


def test_standard_attention_throughput(qkv):
  """Time ``standard_attention`` at LLM scale."""
  q, k, v = qkv
  out = _bench("standard_attention", attnax.standard_attention, q, k, v)
  assert out.shape == q.shape


def test_memory_efficient_matches_standard(qkv, reference):
  """``memory_efficient_attention`` agrees with standard at LLM scale."""
  q, k, v = qkv
  out = _bench(
    "memory_efficient_attention (block=128)",
    lambda q_, k_, v_: attnax.memory_efficient_attention(
      q_, k_, v_, block_size=128
    ),
    q, k, v,
  )
  np.testing.assert_allclose(out, reference, atol=1e-3, rtol=1e-3)


def test_flash_attention_matches_standard(qkv, reference):
  """``flash_attention`` (jax.nn-backed) agrees with standard at LLM scale."""
  q, k, v = qkv
  out = _bench("flash_attention", attnax.flash_attention, q, k, v)
  np.testing.assert_allclose(out, reference, atol=1e-3, rtol=1e-3)


def test_pallas_flash_matches_standard(qkv, reference):
  """``pallas_flash_attention`` agrees with standard at LLM scale."""
  q, k, v = qkv
  out = _bench(
    "pallas_flash_attention", attnax.pallas_flash_attention, q, k, v
  )
  np.testing.assert_allclose(out, reference, atol=1e-3, rtol=1e-3)


def test_transformer_encoder_forward():
  """Forward pass through a multi-layer encoder at LLM scale."""
  config = attnax.TransformerConfig(
    vocab_size=_VOCAB,
    d_model=_D_MODEL,
    num_heads=_NUM_HEADS,
    num_layers=_NUM_LAYERS,
    d_ff=_D_FF,
    max_len=_SEQ,
    dropout_rate=0.0,
  )
  model = attnax.TransformerEncoder(nnx.Rngs(0), config)
  ids = jax.random.randint(
    jax.random.key(1), (_BATCH, _SEQ), 0, _VOCAB
  )

  def fwd(ids_):
    return model(ids_, deterministic=True)

  out = _bench(
    f"TransformerEncoder ({_NUM_LAYERS}L, d_model={_D_MODEL}, seq={_SEQ})",
    fwd, ids,
  )
  assert out.shape == (_BATCH, _SEQ, _D_MODEL)


def test_decoder_kv_cache_inference():
  """Autoregressive decode with KV cache at LLM scale."""
  config = attnax.TransformerConfig(
    vocab_size=_VOCAB,
    d_model=_D_MODEL,
    num_heads=_NUM_HEADS,
    num_layers=_NUM_LAYERS,
    d_ff=_D_FF,
    max_len=_SEQ,
    dropout_rate=0.0,
  )
  model = attnax.TransformerDecoder(nnx.Rngs(0), config)

  prompt_len = 64
  decode_steps = 32
  prompt = jax.random.randint(
    jax.random.key(2), (_BATCH, prompt_len), 0, _VOCAB
  )

  caches = attnax.init_decoder_kv_caches_from_config(
    config, batch_size=_BATCH, max_len=_SEQ
  )

  # Prefill: encode the prompt and fill the KV cache.
  start = time.perf_counter()
  prefill_out, caches = model(
    prompt, layer_kv_caches=caches, deterministic=True
  )
  jax.block_until_ready(prefill_out)
  prefill_ms = (time.perf_counter() - start) * 1000
  print(f"  decoder prefill (prompt_len={prompt_len}): {prefill_ms:.2f} ms")
  assert prefill_out.shape == (_BATCH, prompt_len, _D_MODEL)

  # Decode loop: one token at a time.
  next_tok = jax.random.randint(jax.random.key(3), (_BATCH, 1), 0, _VOCAB)
  # Warmup the per-step trace.
  step_out, caches = model(
    next_tok, layer_kv_caches=caches, deterministic=True
  )
  jax.block_until_ready(step_out)

  start = time.perf_counter()
  for _ in range(decode_steps):
    step_out, caches = model(
      next_tok, layer_kv_caches=caches, deterministic=True
    )
  jax.block_until_ready(step_out)
  total_ms = (time.perf_counter() - start) * 1000
  print(
    f"  decoder decode loop ({decode_steps} steps): "
    f"{total_ms:.2f} ms total, {total_ms / decode_steps:.2f} ms/step"
  )
  assert step_out.shape == (_BATCH, 1, _D_MODEL)


def test_vision_transformer_forward():
  """ViT-Base/16 forward pass on ImageNet-scale images."""
  config = attnax.VisionTransformerConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    num_classes=1000,
    d_model=768,
    num_heads=12,
    num_layers=_NUM_LAYERS,
    d_ff=3072,
    dropout_rate=0.0,
  )
  model = attnax.VisionTransformer(nnx.Rngs(0), config)
  images = jax.random.normal(jax.random.key(4), (16, 224, 224, 3))

  def fwd(images_):
    return model(images_, deterministic=True)

  out = _bench(
    f"VisionTransformer (B/16, {_NUM_LAYERS}L, batch=16)", fwd, images
  )
  assert out.shape == (16, 1000)
