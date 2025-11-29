# Ultra Fast Image Generation for Mac Silicon (Z-Image)

6B parameter AI image generation that actually runs fast on your Mac. No cloud. No GPU rental.

## Features

- **Ultra Fast**: 14 seconds for 512x512, 31 seconds for 768x768
- **Tiny Model**: Only 3.5GB (4-bit quantized from 24GB)
- **Runs Locally**: No API keys, no cloud costs
- **Gradio UI**: Simple web interface included
- **CUDA Support**: Experimental support for NVIDIA GPUs

## Benchmarks

### Mac M1 Max (64GB)

| Resolution | Steps | Time | Speed |
|------------|-------|------|-------|
| 512x512 | 7 | **23s** | 3.38s/step |
| 768x768 | 7 | **49s** | 7.13s/step |

### Mac M2 Max (32GB)

| Resolution | Steps | Time | Speed |
|------------|-------|------|-------|
| 512x512 | 7 | **14s** | 2.08s/step |
| 768x768 | 7 | **31s** | 4.43s/step |

### Mac M3 (18GB)

| Resolution | Steps | Time | Speed |
|------------|-------|------|-------|
| 512x512 | 7 | **74s** | 10.6s/step |
| 768x768 | 7 | **2.5 min** | 21.4s/step |

**RAM Usage**: ~15.5 GB

**Recommended**: 768x768 @ 7-8 steps for best quality/speed balance.

> **Note**: M2 Max has significantly more GPU cores (30-38) vs base M3 (10), which explains the speed difference. M3 Pro/Max should be much faster.

## Model

Uses [Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) - a 4-bit quantized version of Alibaba's [Z-Image Turbo](https://github.com/Tongyi-MAI/Z-Image) (6B parameter diffusion transformer).

## Quick Start (1-Click)

1. Download/clone the repo
2. **Double-click `Launch.command`**
3. First run will auto-install dependencies (~5 min)
4. Browser opens automatically to the UI

That's it! The launcher handles everything.

## Manual Installation

```bash
# Clone the repo
git clone https://github.com/newideas99/Ultra-Fast-Image-Generation-Mac-Silicon-Z-Image.git
cd Ultra-Fast-Image-Generation-Mac-Silicon-Z-Image

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Gradio Web UI (Easiest)

Double-click `Launch.command` or run:

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

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

Example:
```bash
python generate.py "Cyberpunk city at night, neon lights" --height 768 --width 768 --steps 7 --seed 42 --output cyberpunk.png
```

## Performance Tips

- **Resolution**: 512x512 is fastest. 768x768 is good quality/speed balance.
- **Steps**: 5 steps is usually enough for good results. More steps = slower.
- **First run**: First generation is slower due to model loading (~30-60s). Subsequent generations are faster.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- **16GB+ unified memory required** (~15.5GB used during inference)

## Credits

- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) by Alibaba
- [SDNQ Quantization](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) by Disty0

## License

See the original model licenses for usage terms.
