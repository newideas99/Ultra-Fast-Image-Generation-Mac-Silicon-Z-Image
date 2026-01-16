# Ultra Fast Image Gen

AI image generation and editing on Mac Silicon and CUDA. Generate images from text or transform existing images with state-of-the-art diffusion models.

## Features

- **Image Generation:** Create images from text prompts
- **Image Editing:** Upload up to 6 reference images and transform them with natural language
- **Multiple Models:** FLUX.2-klein and Z-Image Turbo
- **Quantized Models:** Low memory usage with 4bit/int8 quantization
- **LoRA Support:** Load custom LoRA adapters with Z-Image Full model
- **Cross-Platform:** Apple Silicon (MPS) and NVIDIA GPUs (CUDA)

## Supported Models

| Model | VRAM | Features | Speed |
|-------|------|----------|-------|
| FLUX.2-klein-4B (4bit SDNQ) | <8GB @ 512px, <16GB @ 1024px | Text-to-image + Image editing | Fast |
| FLUX.2-klein-9B (4bit SDNQ) | ~12GB @ 512px, ~20GB @ 1024px | Text-to-image + Image editing (Higher Quality) | Fast |
| FLUX.2-klein-4B (Int8) | ~16GB | Text-to-image + Image editing | Fast |
| Z-Image Turbo (Quantized) | ~8GB | Text-to-image | Fastest |
| Z-Image Turbo (Full) | ~24GB | Text-to-image + LoRA | Slower |

## Quick Start (1-Click)

1. Download/clone the repo
2. **Double-click `Launch.command`**
3. First run will auto-install dependencies (~5 min)
4. Browser opens automatically to the UI

## Manual Installation

```bash
git clone https://github.com/newideas99/ultra-fast-image-gen.git
cd ultra-fast-image-gen

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Web UI

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### Model Selection

- **FLUX.2-klein-4B (4bit SDNQ):** Default. Lowest memory, supports image editing
- **FLUX.2-klein-9B (4bit SDNQ):** Higher quality 9B model, more memory
- **FLUX.2-klein-4B (Int8):** Alternative quantization, more memory
- **Z-Image Turbo (Quantized):** Fastest text-to-image, no image editing
- **Z-Image Turbo (Full):** Use when you need LoRA support

### Image Editing (FLUX.2-klein)

1. Select a FLUX.2-klein model from the dropdown (default)
2. Upload up to 6 images in the gallery
3. Write a prompt describing the changes you want
4. Select output resolution (1024px, 1280px, or 1536px)
5. Click Generate

### Command Line

```bash
python generate.py "A beautiful sunset over mountains"
```

Options:
- `--height`: Image height (default: 512)
- `--width`: Image width (default: 512)
- `--steps`: Inference steps (default: 5)
- `--seed`: Random seed (-1 for random)
- `--output`: Output file path (default: output.png)
- `--lora`: Path to LoRA safetensors file
- `--lora-strength`: LoRA strength multiplier (default: 1.0)

## Benchmarks

### FLUX.2-klein-4B

| Hardware | Resolution | Steps | Time |
|----------|------------|-------|------|
| Apple Silicon | 512x512 | 4 | ~8s |
| CUDA (RTX 3090) | 512x512 | 4 | ~3s |

### Z-Image Turbo (Quantized)

| Mac | Resolution | Steps | Time |
|-----|------------|-------|------|
| M2 Max | 512x512 | 7 | 14s |
| M2 Max | 768x768 | 7 | 31s |
| M1 Max | 512x512 | 7 | 23s |

## Memory Requirements

| Model | RAM/VRAM Required |
|-------|-------------------|
| FLUX.2-klein-4B (4bit SDNQ) | 8GB @ 512px, 16GB @ 1024px |
| FLUX.2-klein-9B (4bit SDNQ) | 12GB @ 512px, 20GB @ 1024px |
| FLUX.2-klein-4B (Int8) | 16GB |
| Z-Image (Quantized) | 8GB |
| Z-Image (Full) | 24GB+ |

## Credits

- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) by Alibaba
- [SDNQ Quantization](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic) by Disty0
- [Int8 Quantization](https://huggingface.co/aydin99/FLUX.2-klein-4B-int8) using optimum-quanto

## License

See the original model licenses for usage terms.
