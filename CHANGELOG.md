# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-25

### Added

- Pluggable attention kernels as pure JAX functions sharing a single
  `AttentionFn` protocol: `standard_attention`, `memory_efficient_attention`,
  `flash_attention`, `linear_attention`, `ring_attention`,
  `pallas_flash_attention`, `paged_attention`, `lite_attention`.
- `ScoreMod` and `MaskMod` protocols with constructors for `alibi_mod`,
  `causal_mod`, `sliding_window_mod`, `prefix_lm_mod`, `document_mask_mod`,
  `additive_bias_mod`, and `compose_score_mods`.
- Grouped-query and multi-query attention (`num_kv_heads`) in
  `MultiHeadAttention`.
- Rotary positional embeddings: `RotaryEmbedding`, `apply_rope`,
  `rope_cos_sin_table`, `rope_cos_sin_from_positions`, `rope_inv_freq`.
- KV caches for autoregressive inference and batched serving:
  `KVLayerCache` (contiguous) and `PagedKVCache` (vLLM-style block table)
  with `init_kv_layer_cache`, `update_kv_layer_cache`,
  `init_decoder_kv_caches`, `init_decoder_kv_caches_from_config`,
  `init_paged_kv_cache`, `allocate_blocks`, `append_kv`, `gather_kv`.
- `MixtureOfExperts` feed-forward with top-k routing and a Switch-Transformer
  load-balance loss.
- Gated feed-forward activations: `swiglu` and `geglu`.
- `TransformerDecoder` with built-in causal masking and KV-cache integration.
- Vision Transformer: `VisionTransformer`, `VisionTransformerConfig`,
  `PatchEmbedding`, `LearnedPositionalEmbedding`.
- `RMSNorm` and `create_norm` factory; selectable per layer via
  `TransformerConfig.norm_type`.
- Additional masking utilities: `make_sliding_window_mask`,
  `make_document_mask`.
- Sphinx documentation under `docs/`, hosted on Read the Docs, with a
  getting-started notebook covering kernels, score-mods, KV caching, paged
  caching, MoE, ViT, and training.

### Changed

- `MultiHeadAttentionLayer` renamed to `MultiHeadAttention`. The kernel is
  now chosen by passing any callable matching `AttentionFn` via
  `attention_fn`, or by name via the extended `AttentionType` enum.
- `FeedForward`'s `activation=` parameter renamed to `ff_activation=`;
  accepted values extended from `{"relu", "gelu"}` to
  `{"relu", "gelu", "swiglu", "geglu"}`.
- `TransformerConfig` extended with `ff_activation`, `norm_type`,
  `num_kv_heads`, `attention_window`, `linear_attention_chunk_size`,
  `pos_emb_type`, `rope_base`, `rope_max_positions`.
- `EncoderBlock` and `DecoderBlock` accept the new options for norm,
  activation, GQA, RoPE, KV cache, and score-mod.
- Minimum Python bumped to 3.10. Minimum JAX bumped to 0.10.0,
  Flax to 0.12.7.

### Removed

- `attnax/encoder.py` (superseded by `attnax/transformer.py`, which provides
  both `TransformerEncoder` and `TransformerDecoder`).
- `examples/minimal_encoder.py` (superseded by the getting-started
  notebook).

## [0.1.0] - 2025-11-15

### Added

- Initial release of Attnax.
- `TransformerEncoder` with configurable depth and width.
- `EncoderBlock` and `DecoderBlock` with multi-head attention and a
  position-wise feed-forward network.
- `MultiHeadAttentionLayer` supporting self- and cross-attention, with
  `standard`, `memory_efficient`, `flash`, and `lite` backends.
- `FeedForward` with `relu` and `gelu` activations.
- `TokenEmbedding` and sinusoidal `PositionalEncoding`.
- Masking utilities: `make_padding_mask`, `make_causal_mask`,
  `combine_masks`.
- `TransformerConfig` dataclass and `AttentionType` enum.

[0.2.0]: https://github.com/glibtkachenko/attnax/releases/tag/v0.2.0
[0.1.0]: https://github.com/glibtkachenko/attnax/releases/tag/v0.1.0
