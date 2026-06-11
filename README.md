# Ultra Fast Image Gen MCP

A high-performance, local Model Context Protocol (MCP) server that empowers AI coding assistants (like OpenCode, Claude Desktop, and Cursor) to generate and edit images directly within your project workflow. 

![Ultra Fast Image Gen web UI](docs/screenshot.png)

Eliminate API costs, maintain data privacy, and leverage state-of-the-art diffusion models (FLUX.2-klein, Z-Image Turbo, Anima) with optimized quantization for Apple Silicon and NVIDIA GPUs.

---

## 🚀 Key Features

- **Seamless MCP Integration**: AI agents can invoke image generation and editing directly from chat.
- **Local & Private**: All processing happens on your machine. No external API keys or data leaks.
- **Multiple Advanced Models**: Support for FLUX.2-klein (4B/9B), Z-Image Turbo, Anima Turbo AIO, and uncensored 2K lanes.
- **Optimized for Mac/CUDA**: 4-bit and 8-bit quantization ensures low memory footprint without sacrificing quality.
- **Image-to-Image Editing**: Transform existing assets using natural language prompts (up to 6 reference images).
- **Bonsai Image 4B**: Ternary FLUX.2 Klein, in-process MLX on Apple Silicon (opt-in).

---

## ⚙️ MCP Server Setup (Recommended)

This project is designed to be used as an MCP server. Choose your preferred AI client below.

### 1. OpenCode (1-Click Installation)

Run the provided installation script to surgically inject the MCP server configuration and visual assets skill into your global OpenCode config (`~/.config/opencode/opencode.json`):

```bash
python3 scripts/install-opencode-mcp.py
```
> **Note:** The script will prompt you for your Hugging Face token (`HF_TOKEN`) if it is not already set in your environment. This token is required for downloading model weights.

### 2. Manual Configuration

If you prefer to configure manually, add the following to your client's configuration file. Ensure you replace `/path/to/ultra-fast-image-gen-mcp` with your actual repository path.

**OpenCode (`~/.config/opencode/opencode.json` or project `opencode.json`):**
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "ultra-fast-image-gen": {
      "type": "local",
      "command": ["python3", "/path/to/ultra-fast-image-gen-mcp/mcp_server.py"],
      "environment": {
        "HF_TOKEN": "your_huggingface_token_here"
      },
      "timeout": 3600000
    }
  },
  "experimental": {
    "mcp_timeout": 3600000
  }
}
```

**Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "ultra-fast-image-gen": {
      "command": "python3",
      "args": ["/path/to/ultra-fast-image-gen-mcp/mcp_server.py"],
      "env": {
        "HF_TOKEN": "your_huggingface_token_here"
      }
    }
  }
}
```

### 3. Agent Instructions

To ensure your AI agent uses the tools effectively, add this to your project's `.opencode/agents.md`, `CLAUDE.md`, or `.cursorrules`:

```markdown
## Image Generation & Editing
When asked to create or modify visual assets (e.g., "generate a hero banner", "make the logo background dark"), use the `generate_image` or `edit_image` tools from the `ultra-fast-image-gen` MCP server. 
- Default model is `zimage-quant` (ultra-fast, lowest memory). 
- Ask for `model="flux2-4b-sdnq"` for high quality, or `model="flux2-9b-sdnq"` for highest quality.
- Always save outputs to logical project paths (e.g., `public/images/hero.png` or `src/assets/icon.svg`).
- For image editing, provide 1-6 existing image paths in the `input_image_paths` array.
```

### 4. No Environment Variables Needed

The MCP server works out of the box — no `HF_TOKEN` or other environment variables are required for image generation. Model weights are downloaded once to the local Hugging Face cache and reused on subsequent runs.

> **Note:** `HF_TOKEN` is only needed if you're accessing gated models (like the uncensored text encoder). For standard FLUX.2-klein, Z-Image Turbo, and Anima models, no token is required.

---

## 🧠 Supported Models

| Model | VRAM (Approx.) | Features | Best For |
|-------|----------------|----------|----------|
| **FLUX.2-klein-4B (4bit SDNQ)** | <8GB @ 512px, <16GB @ 1024px | Text-to-Image, Image Editing | Balanced speed & quality |
| **FLUX.2-klein-9B (4bit SDNQ)** | ~12GB @ 512px, ~20GB @ 1024px | Text-to-Image, Image Editing | Highest quality generation |
| **FLUX.2-klein-4B Uncensored MFLUX HS** | ~7GB RSS @ 2048px | Text-to-Image (2K), Uncensored GGUF TE | Fastest current 2K FLUX lane |
| **FLUX.2-klein-4B Uncensored SDNQ HS** | Low MPS memory @ 2048px | Text-to-Image (2K), Uncensored GGUF TE | PyTorch 2K fallback |
| **FLUX.2-klein-4B (Int8)** | ~16GB | Text-to-Image, Image Editing | Alternative quantization |
| **Z-Image Turbo (Quantized)** | ~3.5GB | Text-to-Image only | Ultra-fast prototyping |
| **Anima Turbo AIO Q4 (Metal)**| ~3GB model + unified memory | Text-to-Image (Baked LoRA) | Apple Silicon optimized |
| **Z-Image Turbo (Full)** | ~24GB+ | Text-to-Image + Custom LoRA | Advanced fine-tuning |
| **Bonsai Image 4B (Ternary MLX)**| ~3.7GB, Apple Silicon only | Text-to-Image, 4 steps | Ultra-low memory MLX |

> **Note:** The [uncensored text encoder repo](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder) is gated on Hugging Face (instant auto-approval). Accept the terms on the model page once, then set `HF_TOKEN=hf_...` in your `.env` file.

---

## 💻 Traditional Usage (Web UI & CLI)

If you prefer to run the application standalone without an AI agent, you can use the Web UI or the Command Line Interface.

### Quick Start (1-Click)
1. Clone or download the repository.
2. **Double-click `Launch.command`** (macOS).
3. The script will automatically create a virtual environment, install dependencies, and build the Anima Metal runner if needed.
4. Your browser will open to `http://127.0.0.1:7860`.

### Manual Installation
```bash
git clone https://github.com/newideas99/ultra-fast-image-gen.git
cd ultra-fast-image-gen

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# For the uncensored models (gated text-encoder repo):
echo "HF_TOKEN=hf_your_token_here" > .env
```

### Web UI
```bash
python server.py
```

### Command Line Interface
Each model has a dedicated sub-command. Quotes around the prompt are optional; all words before the first `--flag` are treated as the prompt.

```bash
# Z-Image Turbo (Quantized) — Fastest
python generate.py zimage-quant a beautiful sunset over mountains

# FLUX.2-klein-9B (4bit SDNQ) — Highest Quality
python generate.py flux2-9b-sdnq a cyberpunk cityscape at night --guidance 3.5 --steps 28

# FLUX.2-klein-4B Uncensored MFLUX/MLX HS — fastest current 2K path
python generate.py flux2-4b-uncensored-mflux-hs a beautiful sunset --width 2048 --height 2048 --steps 4

# Image-to-Image Editing (FLUX models only)
python generate.py flux2-4b-sdnq transform the fox into a wolf --input-images ref.png

# Anima Turbo AIO (Metal runner)
python generate.py anima anime portrait, detailed eyes --anima-preset Balanced

# Bonsai Image 4B ternary (MLX, Apple Silicon) — needs the bonsai extra
python generate.py bonsai-ternary a red fox in snow --steps 4
```

#### CLI Options Reference
| Option | Default | Description |
|--------|---------|-------------|
| `--height` | 512 (Anima: 768) | Image height in pixels |
| `--width` | 512 | Image width in pixels |
| `--seed` | random | Fixed seed for reproducibility |
| `--output` | output.png | Output file path |
| `--device` | mps | `mps`, `cuda`, or `cpu` |
| `--steps` | 5 (Z-Image) / 28 (FLUX) | Inference steps |
| `--guidance` | 3.5 | Classifier-free guidance scale (FLUX) |
| `--input-images` | — | Up to 6 reference images for editing |

**Uncensored 2K speed-lane options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 4 | Distilled klein inference steps |
| `--guidance` | 0.0 | Guidance scale |
| `--gguf-quant` | q4_k_m | Uncensored Qwen GGUF text encoder quant |
| `--qchunk` | 1024 | PyTorch MPS attention query chunk (`sdnq-hs` only) |
| `--hs-stride` | 2 | Hidden-state compression stride |
| `--hs-max-transformer-forward` | steps - 1 | Leave the final transformer forward exact |

**Anima options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | from preset | Inference steps (overrides the preset) |
| `--cfg-scale` | 1.0 | Anima CFG scale |
| `--anima-preset` | Balanced | `Fast` (3 steps), `Balanced` (8), or `Quality` (16) |

**Bonsai options** (Apple Silicon only; `--device` is accepted but ignored — always MLX):
| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 4 | Inference steps |

---

## 📊 Performance & Benchmarks

### FLUX.2-klein-4B
| Hardware | Resolution | Steps | Time |
|----------|------------|-------|------|
| Apple Silicon (MPS) | 512x512 | 4 | ~8s |
| NVIDIA CUDA (RTX 3090) | 512x512 | 4 | ~3s |

### FLUX.2-klein-4B Uncensored 2K Speed Lanes
| Backend | Resolution | Steps | Time | Notes |
|---------|------------|-------|------|-------|
| MFLUX/MLX HS | 2048x2048 | 4 | 100.2s fresh-process wall / ~69s denoise | Includes uncensored GGUF TE load in fresh CLI process |
| PyTorch SDNQ HS | 2048x2048 | 4 | ~110s generation wall | Exact query-chunked MPS attention + HS compression |

### Z-Image Turbo (Quantized)
| Hardware | Resolution | Steps | Time |
|----------|------------|-------|------|
| Apple M2 Max | 512x512 | 7 | 14s |
| Apple M2 Max | 768x768 | 7 | 31s |
| Apple M1 Max | 512x512 | 7 | 23s |

### Anima Turbo AIO Q4 (Metal)
| Hardware | Resolution | Steps | Time (Internal / Wall) |
|----------|------------|-------|------------------------|
| Apple M2 Max | 512x768 | 3 | 8.82s / 11.63s |
| Apple M2 Max | 512x768 | 8 | 15.62s / 18.69s |

> **Note:** First-time execution will download model weights (up to 12.6GB for FLUX 9B). Subsequent runs will load from the local Hugging Face cache instantly.

---

## 🛠️ Credits & Acknowledgments

- **FLUX.2-klein**: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- **Z-Image**: [Alibaba Tongyi-MAI](https://github.com/Tongyi-MAI/Z-Image)
- **Anima**: [Circlestone Labs](https://huggingface.co/circlestone-labs/Anima)
- **SDNQ Quantization**: [Disty0](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic)
- **Int8 Quantization**: [aydin99](https://huggingface.co/aydin99/FLUX.2-klein-4B-int8) using `optimum-quanto`
- **Bonsai Image 4B**: [PrismML](https://github.com/PrismML-Eng/image-studio) (via prism-image-studio + mflux-prism)

---

## 📜 License

This project serves as an interface wrapper. Please refer to the original model licenses (Black Forest Labs, Alibaba, Circlestone Labs) for specific usage terms and commercial restrictions.