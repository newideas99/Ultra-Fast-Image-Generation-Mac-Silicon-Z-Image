# Benchmarks

Performance metrics for supported models across different hardware configurations.

## FLUX.2-klein-4B

| Hardware | Resolution | Steps | Time |
| :--- | :--- | :--- | :--- |
| Apple Silicon | 512x512 | 4 | ~8s |
| CUDA (RTX 3090) | 512x512 | 4 | ~3s |

## FLUX.2-klein-4B Uncensored 2K Speed Lanes

| Backend | Resolution | Steps | Time | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MFLUX/MLX HS** | 2048x2048 | 4 | 100.2s fresh-process wall / ~69s denoise | Includes uncensored GGUF TE load in fresh CLI process |
| **PyTorch SDNQ HS** | 2048x2048 | 4 | ~110s generation wall | Exact query-chunked MPS attention + HS compression |

## Z-Image Turbo (Quantized)

| Mac | Resolution | Steps | Time |
| :--- | :--- | :--- | :--- |
| M2 Max | 512x512 | 7 | 14s |
| M2 Max | 768x768 | 7 | 31s |
| M1 Max | 512x512 | 7 | 23s |

## Anima Turbo AIO Q4 (Metal)

Recommended settings:
- **Fast:** 3 steps with Spectrum cache
- **Balanced (default):** 8 steps with Spectrum cache
- **Quality (model-card default):** 16 steps with cache disabled

| Mac | Resolution | Steps | Time |
| :--- | :--- | :--- | :--- |
| M2 Max | 512x768 | 3 | 8.82s internal / 11.63s wall |
| M2 Max | 512x768 | 4 | 11.13s internal / 13.65s wall |
| M2 Max | 512x768 | 8 | 15.62s internal / 18.69s wall |

---

**Next:** [Credits & License](/guide/credits)
