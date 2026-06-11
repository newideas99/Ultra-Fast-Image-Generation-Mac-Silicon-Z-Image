# Supported Models

Ultra Fast Image Gen supports multiple state-of-the-art diffusion models, optimized for different hardware configurations and use cases.

## Model Comparison

| Model | VRAM / RAM | Features | Speed |
| :--- | :--- | :--- | :--- |
| **FLUX.2-klein-4B Uncensored MFLUX HS** | ~7GB RSS @ 2K | Text-to-image, uncensored GGUF TE, validated to 2K | Fastest 2K |
| **FLUX.2-klein-4B Uncensored SDNQ HS** | ~8GB MPS / low RAM @ 2K | Text-to-image, exact chunked MPS attention + HS | PyTorch 2K Fallback |
| **FLUX.2-klein-4B (4bit SDNQ)** | <8GB @ 512px, <16GB @ 1024px | Text-to-image + Image editing (up to 6 images) | Fast |
| **FLUX.2-klein-9B (4bit SDNQ)** | ~12GB @ 512px, ~20GB @ 1024px | Text-to-image + Image editing (Higher Quality) | Fast |
| **FLUX.2-klein-4B (Int8)** | ~16GB | Text-to-image + Image editing | Fast |
| **Z-Image Turbo (Quantized)** | ~8GB | Text-to-image only | Fastest |
| **Anima Turbo AIO Q4 (Metal)** | ~3GB model + unified memory | Text-to-image, baked Turbo LoRA | ~16s @ 512x768 / 8 steps |
| **Bonsai Image 4B (Ternary MLX)** | ~3.7GB (Apple Silicon only) | Text-to-image, 4 steps | ~15s @ 512x512 / 4 steps |
| **Z-Image Turbo (Full)** | ~24GB+ | Text-to-image + Custom LoRA support | Slower |

## Important Notes

> **Uncensored Variants:** The uncensored variants do not re-download a separate base model. The SDNQ HS lane and the plain uncensored model reuse the FLUX.2-klein-4B (4bit SDNQ) backbone. The only extra download is the uncensored Qwen3 text encoder (~2.5GB GGUF at the default `q4_k_m` quant).

> **Gated Repository:** The [uncensored text encoder repo](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder) is gated on Hugging Face (instant auto-approval). Accept the terms on the model page once, then paste a token in the app's **⋯ menu**.

> **Bonsai Image 4B:** This is **not** part of the default install. It is an opt-in extra (`uv sync --extra bonsai`) that runs in-process on MLX and is **Apple Silicon + Python 3.11+ only**.

---

**Next:** [Web UI & CLI Usage](/guide/usage)
