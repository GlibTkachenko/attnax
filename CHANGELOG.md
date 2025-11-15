# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-14

### Added
- Initial release of JAXTransformer
- `TransformerEncoder` with configurable layers
- `EncoderBlock` with multi-head attention and feed-forward network
- `MultiHeadAttentionLayer` supporting self and cross-attention
- `FeedForward` position-wise network with configurable activation
- `TokenEmbedding` and `PositionalEncoding` (sinusoidal)
- Masking utilities: `make_padding_mask`, `make_causal_mask`, `combine_masks`
- `TransformerConfig` dataclass for model configuration
- Comprehensive test suite
- Example scripts demonstrating usage
- Full type annotations throughout
- Apache 2.0 license

### Features
- Full JAX transformation support (jit, vmap, grad)
- Composable modules for custom architectures
- Support for variable sequence lengths with padding masks
- Dropout and layer normalization
- Configurable activation functions (GELU, ReLU, Swish)

[0.1.0]: https://github.com/yourusername/jaxtransformer/releases/tag/v0.1.0
