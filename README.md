# Fast Flux Studio

Ultra-fast AI image generation and editing on Mac Silicon and CUDA. Generate images from text or transform existing images with state-of-the-art diffusion models.

## Features

- **Image Generation:** Create images from text prompts
- **Image Editing:** Upload and transform images with natural language (FLUX.2-klein)
- **Multiple Models:** Z-Image Turbo (fastest) and FLUX.2-klein-4B (editing)
- **Quantized Models:** Low memory usage with int4/int8 quantization
- **LoRA Support:** Load custom LoRA adapters with Z-Image Full model
- **Cross-Platform:** Apple Silicon (MPS) and NVIDIA GPUs (CUDA)

## Supported Models

| Model | Size | Features | Speed |
|-------|------|----------|-------|
| FLUX.2-klein-4B (Int8) | 8GB | Text-to-image + Image editing | Fast |
| Z-Image Turbo (Quantized) | 3.5GB | Text-to-image | Fastest |
| Z-Image Turbo (Full) | 24GB | Text-to-image + LoRA | Slower |

## Quick Start (1-Click)

1. Download/clone the repo
2. **Double-click `Launch.command`**
3. First run will auto-install dependencies (~5 min)
4. Browser opens automatically to the UI

## Manual Installation

```bash
git clone https://github.com/newideas99/fast-flux-studio.git
cd fast-flux-studio

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

In the UI, select your model from the dropdown:

- **FLUX.2-klein-4B (Int8):** Default. Best for image editing and high-quality generation
- **Z-Image Turbo (Quantized):** Fastest text-to-image, minimal memory
- **Z-Image Turbo (Full):** Use when you need LoRA support

### Image Editing (FLUX.2-klein)

1. Select "FLUX.2-klein-4B (Int8)" from the model dropdown (default)
2. Upload an image in the "Input Image" section
3. Write a prompt describing the changes you want
4. Adjust the "Strength" slider:
   - Lower values (0.3-0.5): Subtle changes, keeps original structure
   - Higher values (0.7-1.0): More dramatic changes
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

### FLUX.2-klein-4B (Int8)

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
| FLUX.2-klein (Int8) | 16GB |
| Z-Image (Quantized) | 16GB |
| Z-Image (Full) | 32GB+ |

## Credits

- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) by Alibaba
- [SDNQ Quantization](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) by Disty0
- [Int8 Quantization](https://huggingface.co/aydin99/FLUX.2-klein-4B-int8) using optimum-quanto

## License

See the original model licenses for usage terms.
