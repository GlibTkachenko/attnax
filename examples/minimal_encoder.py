# SPDX-License-Identifier: Apache-2.0

"""Minimal transformer encoder example."""

import jax.numpy as jnp
import flax.nnx as nnx

from attnax import TransformerConfig, TransformerEncoder, make_padding_mask


def main():
  config = TransformerConfig(
    vocab_size=32000,
    d_model=256,
    num_heads=4,
    num_layers=2,
    d_ff=1024,
    pad_token_id=0,
  )
  rngs = nnx.Rngs(0)

  model = TransformerEncoder(rngs, config)

  batch_size, seq_len = 2, 10
  input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
  padding_mask = make_padding_mask(input_ids, pad_token_id=config.pad_token_id)

  output = model(input_ids, padding_mask=padding_mask, deterministic=True)
  print(f"Output shape: {output.shape}")
  print(
    f"Expected: (batch={batch_size}, seq_len={seq_len}, "
    f"d_model={config.d_model})"
  )
  assert output.shape == (batch_size, seq_len, config.d_model)
  print("Inference test passed")

  output_train = model(
    input_ids, padding_mask=padding_mask, deterministic=False
  )
  assert output_train.shape == (batch_size, seq_len, config.d_model)
  print("Training test passed")


if __name__ == "__main__":
  main()
