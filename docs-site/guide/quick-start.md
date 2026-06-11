# Quick Start

Get up and running with Ultra Fast Image Gen in minutes.

## 1-Click Launch (Recommended)

1. Download or clone the repository.
2. **Double-click `Launch.command`**.
   *(If macOS says it's from an unidentified developer, right-click the file → Open → Open)*
3. First run will auto-install dependencies (~5 min). Later runs reinstall only if `requirements.txt` changes.
4. The launcher installs/builds the patched Anima Metal runner and the MFLUX 2K runtime if needed.
5. Your browser will open automatically to the Web UI.
6. Models download from inside the app: each one in the model picker has a Download button with live progress.

> **Note:** For Uncensored models, paste your Hugging Face token in the **⋯ menu**. The gated text encoder requires auto-approval on the [model page](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder) once.

## Manual Installation

If you prefer manual setup or are on Linux/Windows:

```bash
git clone https://github.com/newideas99/ultra-fast-image-gen.git
cd ultra-fast-image-gen

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Set Hugging Face token for uncensored models
echo "HF_TOKEN=hf_your_token_here" > .env
```

## Optional Components Setup

### Anima Fresh Install
The `Launch.command` runs this automatically when the Anima runner is missing. To run manually:
```bash
scripts/setup_anima_metal_runner.sh
```
This clones `stable-diffusion.cpp`, applies the bundled ggml Metal patch for Anima VAE ops, builds `sd-cli` with Metal enabled, and writes the runner script.

### MFLUX HS 2K Lane
Also run automatically by `Launch.command`. To run manually:
```bash
scripts/setup_mflux_hs.sh
```
This clones the patched `mflux` repository and pre-builds the `uv` environment for the fastest 2K generation path.

### Bonsai Image 4B (Apple Silicon Only)
Bonsai is an opt-in extra requiring Python 3.11+ and `uv`:
```bash
uv sync --extra bonsai
```
Weights (~3.7 GB) auto-download from Hugging Face on first use.

---

**Next:** [Supported Models](/guide/models)
