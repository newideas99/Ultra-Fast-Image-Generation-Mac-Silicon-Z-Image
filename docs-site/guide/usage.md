# Web UI & CLI Usage

## Web UI

Start the local FastAPI backend and serve the dependency-free HTML/CSS/JS frontend:

```bash
python server.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Web UI Features
- **Real per-step progress** tracking
- **Batch generation** (up to 8 images per run)
- **Resolution presets** (512px, 768px, 1024px, 1280px, 1536px, 2048px)
- **Drag-and-drop image editing** (up to 6 reference images)
- **Session gallery** and storage management
- **Persistent settings** remembered between visits
- **In-app model downloading** with live progress per model

### Image Editing Workflow (FLUX.2-klein)
1. Select a classic FLUX.2-klein model from the dropdown (4bit SDNQ, 9B, or Int8).
2. Upload up to 6 reference images in the gallery.
3. Write a prompt describing the changes you want.
4. Select output resolution.
5. Click Generate.

---

## Command Line Interface

Each model has its own sub-command with the specific options it requires:

```bash
# Z-Image Turbo (quantized) — fastest, ~3.5 GB
python generate.py zimage-quant "a beautiful sunset over mountains"

# Z-Image Turbo (full precision) — with optional LoRA
python generate.py zimage-full "a beautiful sunset" --lora my.safetensors --lora-strength 0.8

# FLUX.2-klein-4B (4bit SDNQ)
python generate.py flux2-4b-sdnq "a beautiful sunset" --guidance 3.5 --steps 28

# FLUX.2-klein-4B Uncensored MFLUX/MLX HS — fastest current 2K path
python generate.py flux2-4b-uncensored-mflux-hs "a beautiful sunset" --width 2048 --height 2048 --steps 4

# Image-to-image editing (FLUX.2-klein models only)
python generate.py flux2-4b-sdnq "transform the fox into a wolf" --input-images ref.png

# Anima Turbo AIO (Metal runner, baked Turbo LoRA)
python generate.py anima "anime portrait, detailed eyes" --anima-preset Balanced

# Bonsai Image 4B ternary (MLX, Apple Silicon)
python generate.py bonsai-ternary "a red fox in snow" --steps 4
```

> **Note:** Quotes around the prompt are optional — all words before the first `--flag` are joined into the prompt.

### Common Options (All Sub-commands)

| Option | Default | Description |
|--------|---------|-------------|
| `--height` | 512 (Anima: 768) | Image height in pixels |
| `--width` | 512 | Image width in pixels |
| `--seed` | random | Fixed seed for reproducibility |
| `--output` | output.png | Output file path |
| `--device` | mps | `mps`, `cuda`, or `cpu` |

### Model-Specific Options

**Z-Image:**
- `--steps`: Inference steps (default: 5)
- `--lora`: Path to LoRA `.safetensors` (`zimage-full` only)
- `--lora-strength`: LoRA weight (default: 1.0)

**FLUX.2-klein:**
- `--steps`: Inference steps (default: 28)
- `--guidance`: Classifier-free guidance scale (default: 3.5)
- `--input-images`: Up to 6 reference images for editing

**Uncensored 2K Speed-lane:**
- `--steps`: Distilled klein inference steps (default: 4)
- `--guidance`: Guidance scale (default: 0.0)
- `--gguf-quant`: Uncensored Qwen GGUF text encoder quant (default: q4_k_m)
- `--qchunk`: PyTorch MPS attention query chunk (`sdnq-hs` only, default: 1024)
- `--hs-stride`: Hidden-state compression stride (default: 2)

**Anima:**
- `--steps`: Inference steps (overrides the preset)
- `--cfg-scale`: Anima CFG scale (default: 1.0)
- `--anima-preset`: `Fast` (3 steps), `Balanced` (8), or `Quality` (16)

---

**Next:** [MCP Server](/guide/mcp)
