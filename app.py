"""
Flux Image Generator - Gradio Web Interface

Fast image generation on Apple Silicon and CUDA.
Supports multiple models:
- Z-Image Turbo (quantized/full)
- FLUX.2-klein-4B (int8 quantized)

FLUX.2-klein also supports image-to-image editing!
"""

import os
import time
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"   # let MPS use all available unified memory
os.environ["HF_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

import torch
import gradio as gr
from PIL import Image
import json
import atexit
import shutil
import tempfile
from datetime import datetime

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen")


def cleanup_gradio_cache():
    gradio_temp = os.path.join(tempfile.gettempdir(), "gradio")
    if os.path.exists(gradio_temp):
        try:
            shutil.rmtree(gradio_temp)
            print("Cleaned up Gradio cache.")
        except Exception:
            pass

atexit.register(cleanup_gradio_cache)

# Global state
pipe = None
img2img_pipe = None  # cached ZImageImg2ImgPipeline — shared weights with pipe
current_device = None
current_model = None  # "zimage-quant", "zimage-full", "flux2-klein-int8"
current_lora_path = None
model_source = "local"  # "local" or "hf_cache"
last_sync_status = ""
online_version_cache = {}  # repo_id -> sha string, or None if error
upscaler_model_cache = {}  # safetensors path -> loaded spandrel model
video_pipe           = None
current_video_device = None

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")

# Model choices
MODEL_CHOICES = [
    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)",
    "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)",
    "FLUX.2-klein-4B (Int8)",
    "Z-Image Turbo (Quantized - Fast)",
    "Z-Image Turbo (Full - LoRA support)",
    "LTX-Video  (txt2video · img2video with ref)",
]


def get_available_devices():
    """Get list of available devices."""
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


# =============================================================================
# Model Source / Cache Directory Helpers
# =============================================================================

def get_local_models_dir():
    """Project-local model cache (ultra-fast-image-gen-main/models)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def get_hf_global_cache_dir():
    """System-wide HuggingFace cache (~/.cache/huggingface/hub)."""
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def get_active_cache_dir():
    """Return the cache dir to USE for loading, based on model_source."""
    if model_source == "hf_cache":
        return get_hf_global_cache_dir()
    return get_local_models_dir()


# Map UI model choices to their primary HuggingFace repo IDs
MODEL_PRIMARY_REPO = {
    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)":      "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
    "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)": "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
    "FLUX.2-klein-4B (Int8)":                        "aydin99/FLUX.2-klein-4B-int8",
    "Z-Image Turbo (Quantized - Fast)":             "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
    "Z-Image Turbo (Full - LoRA support)":          "Tongyi-MAI/Z-Image-Turbo",
}


def get_model_snapshot_info(cache_dir, repo_id):
    """Return (commit_hash, mtime) for a model in a cache dir, or (None, None) if absent."""
    refs_main = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}", "refs", "main")
    if not os.path.exists(refs_main):
        return None, None
    with open(refs_main) as f:
        commit_hash = f.read().strip()
    return commit_hash, os.path.getmtime(refs_main)


def compare_model_versions(repo_id):
    """
    Compare local (project/models) vs HF global cache versions.
    Returns (status, message) where status is one of:
      'none', 'same', 'hf_only', 'local_only', 'hf_newer', 'local_newer'
    """
    local_hash, local_mtime = get_model_snapshot_info(get_local_models_dir(), repo_id)
    hf_hash, hf_mtime = get_model_snapshot_info(get_hf_global_cache_dir(), repo_id)

    if local_hash is None and hf_hash is None:
        return "none", "Not downloaded"
    if local_hash is None:
        return "hf_only", f"Only in HF Cache ({hf_hash[:8]})"
    if hf_hash is None:
        return "local_only", f"Only local ({local_hash[:8]})"
    if local_hash == hf_hash:
        return "same", f"Same ({local_hash[:8]})"
    if hf_mtime > local_mtime:
        return "hf_newer", f"HF newer ({hf_hash[:8]} vs local {local_hash[:8]})"
    return "local_newer", f"Local newer ({local_hash[:8]} vs HF {hf_hash[:8]})"


def sync_from_hf_cache(repo_id):
    """Copy a model from HF global cache to local project folder. Returns status string."""
    display = KNOWN_MODELS.get(repo_id, repo_id)
    src = os.path.join(get_hf_global_cache_dir(), f"models--{repo_id.replace('/', '--')}")
    dst = os.path.join(get_local_models_dir(), f"models--{repo_id.replace('/', '--')}")

    if not os.path.exists(src):
        return f"{display}: not found in HF Cache"
    try:
        os.makedirs(get_local_models_dir(), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        return f"{display}: synced to local"
    except Exception as e:
        return f"{display}: error — {e}"


def auto_sync_if_hf_newer(model_choice):
    """
    If HF cache has a newer version of the selected model, copy it to local.
    Returns a human-readable status string (may be empty if nothing to do).
    """
    repo_id = MODEL_PRIMARY_REPO.get(model_choice)
    if not repo_id:
        return ""
    status, _ = compare_model_versions(repo_id)
    if status in ("hf_newer", "hf_only"):
        return sync_from_hf_cache(repo_id)
    return ""


def check_model_source_status(model_choice, source_choice):
    """
    Update model_source global and return a status string describing
    the version relationship between local and HF cache for the chosen model.
    """
    global model_source
    model_source = "hf_cache" if "HF Cache" in source_choice else "local"

    repo_id = MODEL_PRIMARY_REPO.get(model_choice)
    if not repo_id:
        return ""

    status, detail = compare_model_versions(repo_id)

    if model_source == "local":
        labels = {
            "none":        "Not downloaded locally yet",
            "same":        f"✓ Local: {detail}",
            "local_only":  f"✓ Local only: {detail}",
            "hf_only":     "Not downloaded locally (available in HF Cache)",
            "hf_newer":    f"⚠ HF Cache is newer — switch to HF Cache to update",
            "local_newer": f"✓ Local is newer: {detail}",
        }
    else:
        labels = {
            "none":        "Not in local or HF Cache",
            "same":        f"✓ Up to date — no sync needed",
            "hf_only":     f"⬇ Only in HF Cache — will copy to local on next load",
            "local_only":  f"✓ Only local (HF Cache not available)",
            "hf_newer":    f"🔄 HF Cache is newer — will copy to local on next load",
            "local_newer": f"✓ Local is newer — no sync needed",
        }
    return labels.get(status, detail)


def get_versions_display():
    """Return a markdown table comparing local vs HF cache for all known models."""
    hf_dir = get_hf_global_cache_dir()
    if not os.path.exists(hf_dir):
        return "_HF Cache not found (`~/.cache/huggingface/hub` does not exist)._"

    lines = [
        "| Model | Local | HF Cache | Status |",
        "|-------|-------|----------|--------|",
    ]
    for repo_id, display_name in KNOWN_MODELS.items():
        local_hash, _ = get_model_snapshot_info(get_local_models_dir(), repo_id)
        hf_hash, _ = get_model_snapshot_info(hf_dir, repo_id)
        local_str = f"`{local_hash[:8]}`" if local_hash else "—"
        hf_str    = f"`{hf_hash[:8]}`"    if hf_hash    else "—"
        status, _ = compare_model_versions(repo_id)
        status_icon = {
            "none":        "—",
            "same":        "✓ Same",
            "hf_only":     "⬇ HF only",
            "local_only":  "📁 Local only",
            "hf_newer":    "🔄 HF newer",
            "local_newer": "📁 Local newer",
        }.get(status, status)
        lines.append(f"| {display_name} | {local_str} | {hf_str} | {status_icon} |")

    return "\n".join(lines)


def sync_all_newer_from_hf():
    """Sync all models where HF cache is newer than local. Returns status string."""
    msgs = []
    for repo_id in KNOWN_MODELS:
        status, _ = compare_model_versions(repo_id)
        if status in ("hf_newer", "hf_only"):
            msgs.append(sync_from_hf_cache(repo_id))
    if not msgs:
        return "All local models are already up to date (or HF Cache not available)."
    return "\n".join(msgs)


# =============================================================================
# Online Version Check (HuggingFace Hub API)
# =============================================================================

def _build_online_table():
    """
    Build the online-vs-local comparison table from the cached online_version_cache dict.
    Returns (markdown_string, list_of_display_names_that_can_be_updated).
    """
    if not online_version_cache:
        return "_Click **Check Latest Versions Online** to see results._", []

    lines = [
        "| Model | Local | Latest Online | Status |",
        "|-------|-------|---------------|--------|",
    ]
    updatable = []

    for repo_id, display_name in KNOWN_MODELS.items():
        local_hash, _ = get_model_snapshot_info(get_local_models_dir(), repo_id)
        local_str = f"`{local_hash[:8]}`" if local_hash else "—"

        if repo_id not in online_version_cache:
            online_str, status = "?", "—"
        elif online_version_cache[repo_id] is None:
            online_str, status = "—", "⚠ Error / not found"
        else:
            oh = online_version_cache[repo_id]
            online_str = f"`{oh[:8]}`"
            if local_hash is None:
                status = "⬇ Not downloaded"
                updatable.append(display_name)
            elif local_hash == oh:
                status = "✓ Up to date"
            else:
                status = "🔄 Update available"
                updatable.append(display_name)

        lines.append(f"| {display_name} | {local_str} | {online_str} | {status} |")

    return "\n".join(lines), updatable


def check_online_versions():
    """
    Query HF Hub for the latest commit hash of every known model.
    Populates online_version_cache and returns (table_markdown, dropdown_update).
    """
    from huggingface_hub import HfApi
    global online_version_cache

    api = HfApi()
    for repo_id in KNOWN_MODELS:
        try:
            info = api.model_info(repo_id)
            online_version_cache[repo_id] = info.sha
        except Exception:
            online_version_cache[repo_id] = None

    table, updatable = _build_online_table()
    return table, gr.update(choices=updatable, value=None)


def download_model_update(model_selection):
    """Download / update a single model from HF Hub into the local cache."""
    from huggingface_hub import snapshot_download

    table, updatable = _build_online_table()   # default return values

    if not model_selection:
        return "No model selected.", table, gr.update(choices=updatable, value=None)

    repo_id = next((rid for rid, dn in KNOWN_MODELS.items() if dn == model_selection), None)
    if not repo_id:
        return f"Unknown model: {model_selection}", table, gr.update(choices=updatable, value=None)

    try:
        print(f"[Online update] Downloading {model_selection}...")
        snapshot_download(repo_id, cache_dir=get_local_models_dir())
        msg = f"✓ {model_selection}: updated successfully"
    except Exception as e:
        msg = f"⚠ {model_selection}: {e}"

    table, updatable = _build_online_table()   # refresh after download
    return msg, table, gr.update(choices=updatable, value=None)


def download_all_online_updates():
    """Download every model that has an update available (per online_version_cache)."""
    from huggingface_hub import snapshot_download

    if not online_version_cache:
        table, updatable = _build_online_table()
        return "Run 'Check Latest Versions Online' first.", table, gr.update(choices=updatable, value=None)

    msgs = []
    for repo_id, online_hash in online_version_cache.items():
        if not online_hash:
            continue
        local_hash, _ = get_model_snapshot_info(get_local_models_dir(), repo_id)
        if local_hash == online_hash:
            continue
        display = KNOWN_MODELS.get(repo_id, repo_id)
        try:
            print(f"[Online update] Downloading {display}...")
            snapshot_download(repo_id, cache_dir=get_local_models_dir())
            msgs.append(f"✓ {display}: updated")
        except Exception as e:
            msgs.append(f"⚠ {display}: {e}")

    table, updatable = _build_online_table()
    if not msgs:
        return "All models are already up to date.", table, gr.update(choices=updatable, value=None)
    return "\n".join(msgs), table, gr.update(choices=updatable, value=None)


def load_zimage_pipeline(device="mps", use_full_model=False):
    """Load Z-Image pipeline (quantized or full)."""
    import sdnq  # Required for quantized model
    from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler

    cache_dir = get_active_cache_dir()

    if use_full_model:
        print(f"Loading Z-Image-Turbo (full precision) on {device}...")
        dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )
    else:
        print(f"Loading Z-Image-Turbo UINT4 (quantized) on {device}...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )

    pipe.to(device)
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    return pipe


def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def print_memory(label):
    """Print memory usage with label."""
    mem = get_memory_usage()
    print(f"  [MEM] {label}: {mem:.2f} GB")


def load_flux2_klein_pipeline(device="mps"):
    """Load FLUX.2-klein-4B with int8 quantized transformer and text encoder."""
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen3ForCausalLM, AutoTokenizer, AutoConfig
    from optimum.quanto import requantize
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from quantized_flux2 import QuantizedFlux2Transformer2DModel

    cache_dir = get_active_cache_dir()

    print(f"Loading FLUX.2-klein-4B (int8 quantized) on {device}...")
    print_memory("Before loading")

    model_path = snapshot_download("aydin99/FLUX.2-klein-4B-int8", cache_dir=cache_dir)

    print("  Loading int8 transformer...")
    qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(model_path)
    qtransformer.to(device=device, dtype=torch.bfloat16)
    print_memory("After transformer")

    print("  Loading int8 text encoder...")
    config = AutoConfig.from_pretrained(f"{model_path}/text_encoder", trust_remote_code=True)
    with init_empty_weights():
        text_encoder = Qwen3ForCausalLM(config)

    with open(f"{model_path}/text_encoder/quanto_qmap.json", "r") as f:
        qmap = json.load(f)
    state_dict = load_file(f"{model_path}/text_encoder/model.safetensors")
    requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
    text_encoder.eval()
    text_encoder.to(device, dtype=torch.bfloat16)
    print_memory("After text encoder")

    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

    print("  Loading VAE and scheduler...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        transformer=None,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    print_memory("After VAE/scheduler download")
    
    pipe.transformer = qtransformer._wrapped
    pipe.text_encoder = text_encoder
    pipe.tokenizer = tokenizer
    pipe.to(device)
    print_memory("After pipe.to(device)")
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")
    
    print("  FLUX.2-klein-4B ready!")
    return pipe


def load_flux2_klein_sdnq_pipeline(device="mps"):
    from sdnq import SDNQConfig
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    cache_dir = get_active_cache_dir()

    print(f"Loading FLUX.2-klein-4B (4bit SDNQ) on {device}...")
    print_memory("Before loading")

    print("  Loading tokenizer from base model (SDNQ model missing vocab files)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        subfolder="tokenizer",
        use_fast=False,
        cache_dir=cache_dir,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    print_memory("After loading")
    
    pipe.to(device)
    print_memory("After pipe.to(device)")
    
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")
    
    print("  FLUX.2-klein-4B (SDNQ) ready!")
    return pipe


def load_flux2_klein_9b_sdnq_pipeline(device="mps"):
    from sdnq import SDNQConfig
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    cache_dir = get_active_cache_dir()

    print(f"Loading FLUX.2-klein-9B (4bit SDNQ) on {device}...")
    print_memory("Before loading")

    print("  Loading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        subfolder="tokenizer",
        use_fast=False,
        cache_dir=cache_dir,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    print_memory("After loading")
    
    pipe.to(device)
    print_memory("After pipe.to(device)")
    
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")
    
    print("  FLUX.2-klein-9B (SDNQ) ready!")
    return pipe


def load_pipeline(model_choice: str, device: str = "mps"):
    global pipe, img2img_pipe, current_device, current_model, current_lora_path, last_sync_status

    # If HF Cache mode, sync to local before loading (only when HF version is newer/missing locally)
    last_sync_status = ""
    if model_source == "hf_cache":
        last_sync_status = auto_sync_if_hf_newer(model_choice)
        if last_sync_status:
            print(f"  [Sync] {last_sync_status}")

    if "Quantized" in model_choice:
        model_type = "zimage-quant"
    elif "Full" in model_choice:
        model_type = "zimage-full"
    elif "9B" in model_choice and "SDNQ" in model_choice:
        model_type = "flux2-klein-9b-sdnq"
    elif "4bit SDNQ" in model_choice:
        model_type = "flux2-klein-sdnq"
    elif "FLUX" in model_choice:
        model_type = "flux2-klein-int8"
    else:
        model_type = "zimage-quant"
    
    if pipe is not None and current_device == device and current_model == model_type:
        return pipe
    
    if pipe is not None:
        print(f"Switching from {current_model} to {model_type}...")
        if img2img_pipe is not None:
            del img2img_pipe
            img2img_pipe = None
        del pipe
        current_lora_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    if model_type == "flux2-klein-int8":
        pipe = load_flux2_klein_pipeline(device)
    elif model_type == "flux2-klein-sdnq":
        pipe = load_flux2_klein_sdnq_pipeline(device)
    elif model_type == "flux2-klein-9b-sdnq":
        pipe = load_flux2_klein_9b_sdnq_pipeline(device)
    elif model_type == "zimage-full":
        pipe = load_zimage_pipeline(device, use_full_model=True)
    else:
        pipe = load_zimage_pipeline(device, use_full_model=False)
    
    current_device = device
    current_model = model_type
    print(f"Pipeline loaded on {device}! (Model: {model_type})")
    return pipe


def load_lora(lora_file, lora_strength: float, device: str):
    """Load or update LoRA adapter (Z-Image full model only)."""
    global current_lora_path, pipe
    
    if current_model != "zimage-full":
        return "LoRA only supported with Z-Image Full model"
    
    if lora_file is None or lora_file == "":
        if current_lora_path is not None:
            print("Unloading current LoRA...")
            pipe.unload_lora_weights()
            current_lora_path = None
        return "No LoRA loaded"
    
    lora_path = lora_file if isinstance(lora_file, str) else lora_file.name
    
    if not os.path.exists(lora_path):
        return f"LoRA file not found: {lora_path}"
    
    if not lora_path.endswith('.safetensors'):
        return "Please select a .safetensors file"
    
    if current_lora_path == lora_path:
        pipe.set_adapters(["default"], adapter_weights=[lora_strength])
        return f"Updated LoRA strength to {lora_strength}"
    
    if current_lora_path is not None:
        print(f"Unloading previous LoRA: {current_lora_path}")
        pipe.unload_lora_weights()
    
    try:
        lora_name = os.path.basename(lora_path)
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[lora_strength])
        current_lora_path = lora_path
        return f"Loaded LoRA: {lora_name} (strength={lora_strength})"
    except Exception as e:
        current_lora_path = None
        return f"Error loading LoRA: {str(e)}"


def update_lora_strength(strength: float):
    """Update the LoRA strength without reloading."""
    global pipe, current_lora_path
    if current_lora_path is not None and pipe is not None:
        try:
            pipe.set_adapters(["default"], adapter_weights=[strength])
            return f"LoRA strength updated to {strength}"
        except Exception as e:
            return f"Error updating strength: {str(e)}"
    return "No LoRA loaded"


# =============================================================================
# LTX-Video pipeline
# =============================================================================

def load_ltx_pipeline(device="mps"):
    """Load LTX-Video for text-to-video (reference image handled at inference time)."""
    from diffusers import LTXPipeline

    cache_dir = get_active_cache_dir()
    print(f"Loading LTX-Video on {device}...")
    print_memory("Before loading")
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
    p = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    p.to(device)
    p.enable_attention_slicing()
    if hasattr(p, "enable_vae_slicing"):
        p.enable_vae_slicing()
    print_memory("After loading")
    print("  LTX-Video ready!")
    return p


def export_frames_to_video(frames, output_path, fps=24):
    """Export a list of PIL images to MP4 via imageio."""
    import imageio
    import numpy as np
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(np.array(frame.convert("RGB")))
    writer.close()
    return output_path


# =============================================================================
# Upscaling (spandrel — supports 4x-UltraSharp and any RRDB/ESRGAN safetensors)
# =============================================================================

def load_upscaler(safetensors_path: str):
    """Load a .safetensors upscaler via spandrel. Models are cached by path."""
    import spandrel
    if safetensors_path not in upscaler_model_cache:
        print(f"[Upscaler] Loading model: {safetensors_path}")
        model = spandrel.ModelLoader().load_from_file(safetensors_path)
        model.eval()
        upscaler_model_cache[safetensors_path] = model
    return upscaler_model_cache[safetensors_path]


def upscale_image(image: Image.Image, safetensors_path: str, device: str) -> Image.Image:
    """Upscale a PIL image using a spandrel-compatible upscaler model."""
    import numpy as np

    model = load_upscaler(safetensors_path)
    model.to(device)

    img_rgb = image.convert("RGB") if image.mode != "RGB" else image
    img_np = np.array(img_rgb).astype("float32") / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    with torch.inference_mode():
        out = model(img_tensor.to(device))

    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = (out_np.clip(0, 1) * 255).astype("uint8")
    result = Image.fromarray(out_np)
    print(f"[Upscaler] {image.size} → {result.size}")
    return result


def generate_image(
    prompt,
    height,
    width,
    steps,
    seed,
    guidance,
    device,
    model_choice,
    model_source_choice,
    input_images,
    lora_file,
    lora_strength,
    img_strength,
    repeat_count,
    auto_save,
    output_dir,
    upscale_enabled,
    upscale_model_path,
    num_frames,
    fps_val,
    progress=gr.Progress(track_tqdm=True),
):
    global pipe, img2img_pipe, video_pipe, current_video_device, model_source

    # Apply model source selection before loading
    model_source = "hf_cache" if "HF Cache" in model_source_choice else "local"

    is_video_model = "LTX-Video" in model_choice

    if is_video_model:
        # Unload image pipeline to free memory
        if pipe is not None:
            import gc
            if img2img_pipe is not None:
                del img2img_pipe
                img2img_pipe = None
            del pipe
            pipe = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
        # Load (or reuse) video pipeline
        if video_pipe is None or current_video_device != device:
            if video_pipe is not None:
                del video_pipe
            video_pipe = load_ltx_pipeline(device)
            current_video_device = device
    else:
        # Unload video pipeline if switching back to image
        if video_pipe is not None:
            import gc
            del video_pipe
            video_pipe = None
            current_video_device = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        if "Z-Image" in model_choice and lora_file is not None and lora_file != "":
            model_choice = "Z-Image Turbo (Full - LoRA support)"
        pipe = load_pipeline(model_choice, device)

    if not is_video_model and current_model == "zimage-full" and lora_file:
        load_lora(lora_file, lora_strength, device)

    # Pre-process reference images once — same size/mode for every iteration
    img_w, img_h = int(width), int(height)
    preprocessed_flux_refs  = None
    preprocessed_zimage_ref = None
    if input_images is not None and len(input_images) > 0:
        if current_model in ("flux2-klein-int8", "flux2-klein-sdnq", "flux2-klein-9b-sdnq"):
            preprocessed_flux_refs = []
            for img_data in input_images[:6]:
                img = img_data[0] if isinstance(img_data, tuple) else img_data
                resized = img.copy().resize((img_w, img_h), Image.LANCZOS)
                if resized.mode != "RGB":
                    resized = resized.convert("RGB")
                preprocessed_flux_refs.append(resized)
            print(f"  Pre-processed {len(preprocessed_flux_refs)} reference image(s) → {img_w}×{img_h}")
        elif current_model == "zimage-full":
            raw = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
            preprocessed_zimage_ref = raw.copy().resize((img_w, img_h), Image.LANCZOS)
            if preprocessed_zimage_ref.mode != "RGB":
                preprocessed_zimage_ref = preprocessed_zimage_ref.convert("RGB")
            print(f"  Pre-processed reference image → {img_w}×{img_h}")
        elif is_video_model:
            raw = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
            preprocessed_zimage_ref = raw.copy().resize((img_w, img_h), Image.LANCZOS)
            if preprocessed_zimage_ref.mode != "RGB":
                preprocessed_zimage_ref = preprocessed_zimage_ref.convert("RGB")
            print(f"  Pre-processed LTX reference image → {img_w}×{img_h}")

    base_seed    = int(seed)
    repeat_count = max(1, int(repeat_count or 1))

    for i in range(repeat_count):
        # Seed: random each iteration when -1, otherwise seed / seed+1 / seed+2 …
        if base_seed == -1:
            current_seed = torch.randint(0, 2**32, (1,)).item()
        else:
            current_seed = base_seed + i

        if device == "cuda":
            generator = torch.Generator("cuda").manual_seed(current_seed)
        elif device == "mps":
            generator = torch.Generator("mps").manual_seed(current_seed)
        else:
            generator = torch.Generator().manual_seed(current_seed)

        print_memory(f"Before generation {i + 1}/{repeat_count}")

        iter_label = f"[{i + 1}/{repeat_count}] " if repeat_count > 1 else ""
        yield None, None, f"{iter_label}Generating…"

        _t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                if current_model in ("flux2-klein-int8", "flux2-klein-sdnq", "flux2-klein-9b-sdnq"):
                    if preprocessed_flux_refs is not None:
                        if hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
                            pipe.vae.disable_tiling()

                        image = pipe(
                            prompt=prompt,
                            image=preprocessed_flux_refs if len(preprocessed_flux_refs) > 1 else preprocessed_flux_refs[0],
                            height=img_h,
                            width=img_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                        ).images[0]

                        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                            pipe.vae.enable_tiling()

                        mode = f"img2img ({len(preprocessed_flux_refs)} ref)"
                        video_frames = None
                    else:
                        image = pipe(
                            prompt=prompt,
                            height=img_h,
                            width=img_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                        ).images[0]
                        mode = "txt2img"
                        video_frames = None
                elif current_model == "zimage-full" and preprocessed_zimage_ref is not None:
                    from diffusers import ZImageImg2ImgPipeline
                    if img2img_pipe is None:
                        print("  Creating img2img pipeline (shared weights, one-time cost)...")
                        img2img_pipe = ZImageImg2ImgPipeline.from_pipe(pipe)
                    image = img2img_pipe(
                        prompt=prompt,
                        image=preprocessed_zimage_ref,
                        strength=float(img_strength),
                        height=int(height),
                        width=int(width),
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        generator=generator,
                    ).images[0]
                    mode = "img2img (ref)"
                    video_frames = None
                elif is_video_model:
                    n_frames = max(9, int(num_frames or 25))
                    # LTX requires num_frames = 8k+1
                    n_frames = ((n_frames - 1) // 8) * 8 + 1
                    # Width/height must be divisible by 32
                    ltx_w = max(256, (img_w // 32) * 32)
                    ltx_h = max(256, (img_h // 32) * 32)
                    if preprocessed_zimage_ref is not None:
                        from diffusers import LTXImageToVideoPipeline
                        i2v = LTXImageToVideoPipeline.from_pipe(video_pipe)
                        result = i2v(
                            prompt=prompt,
                            image=preprocessed_zimage_ref,
                            num_frames=n_frames,
                            height=ltx_h,
                            width=ltx_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                        )
                        mode = "img2video"
                    else:
                        result = video_pipe(
                            prompt=prompt,
                            num_frames=n_frames,
                            height=ltx_h,
                            width=ltx_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                        )
                        mode = "txt2video"
                    video_frames = result.frames[0]
                    image = None
                else:
                    image = pipe(
                        prompt=prompt,
                        height=img_h,
                        width=img_w,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        generator=generator,
                    ).images[0]
                    mode = "txt2img"
                    video_frames = None

        except RuntimeError as e:
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                tips = "Try: lower resolution, fewer frames, or close other apps to free RAM."
                yield None, None, f"Out of memory — {tips}"
                return
            raise

        _elapsed = time.perf_counter() - _t0

        print_memory("After generation")

        # Force memory cleanup
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print_memory("After cache clear")

        lora_name  = os.path.basename(lora_file) if lora_file else None
        lora_info  = f" | LoRA: {lora_name} ({lora_strength})" if lora_name else ""
        cfg_info   = f" | CFG: {guidance}" if guidance > 0 else ""
        sync_note  = f" | Sync: {last_sync_status}" if last_sync_status else ""
        time_info  = f" | Time: {_elapsed:.1f}s"

        model_short = {
            "zimage-quant":        "Z-Image (quant)",
            "zimage-full":         "Z-Image (full)",
            "flux2-klein-int8":    "FLUX.2-klein-4B (int8)",
            "flux2-klein-sdnq":    "FLUX.2-klein-4B (4bit)",
            "flux2-klein-9b-sdnq": "FLUX.2-klein-9B (4bit)",
            "ltx-video":           "LTX-Video",
        }.get(current_model or ("ltx-video" if is_video_model else ""), current_model or "")

        if is_video_model and video_frames is not None:
            # Save video to output dir
            v_fps = max(1, int(fps_val or 24))
            timestamp_v = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            prompt_slug = "".join(c if c.isalnum() else "_" for c in (prompt or "")[:30]).strip("_")
            vname = f"{timestamp_v}_{prompt_slug}.mp4" if prompt_slug else f"{timestamp_v}.mp4"
            vpath = os.path.join(get_output_dir(output_dir), vname)
            export_frames_to_video(video_frames, vpath, fps=v_fps)
            info = (f"{iter_label}Seed: {current_seed} | Model: {model_short} | Mode: {mode}"
                    f" | {len(video_frames)} frames @ {v_fps}fps | Device: {device}{cfg_info}"
                    f"{time_info}{sync_note} | Saved: {vpath}")
            yield None, vpath, info
        else:
            # Optional upscaling before save/display
            upscale_note = ""
            if upscale_enabled and upscale_model_path and os.path.isfile(upscale_model_path):
                try:
                    image = upscale_image(image, upscale_model_path, device)
                    upscale_note = f" | Upscaled to {image.size[0]}×{image.size[1]}"
                except Exception as ue:
                    upscale_note = f" | Upscale failed: {ue}"
                    print(f"[Upscaler] Error: {ue}")

            info = (f"{iter_label}Seed: {current_seed} | Model: {model_short} | Mode: {mode}"
                    f" | Device: {device}{cfg_info}{lora_info}{upscale_note}{time_info}{sync_note}")

            if auto_save:
                save_result = save_image(image, output_dir, prompt)
                info += f" | {save_result}"

            yield image, None, info


def clear_lora():
    """Clear the current LoRA."""
    global current_lora_path, pipe
    if current_lora_path is not None and pipe is not None:
        pipe.unload_lora_weights()
        current_lora_path = None
    return None, "LoRA cleared"


# =============================================================================
# Output/Save Functions
# =============================================================================

def get_output_dir(custom_dir=None):
    """Get output directory, creating if needed."""
    output_dir = custom_dir.strip() if custom_dir and custom_dir.strip() else DEFAULT_OUTPUT_DIR
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def browse_upscaler_file(current_path):
    """Open a native macOS file picker for upscaler model files (.pth, .safetensors)."""
    import subprocess
    try:
        # Use choose file without type restriction so both .pth and .safetensors appear
        script = 'POSIX path of (choose file with prompt "Select upscaler model (.pth or .safetensors):")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            return path if path else current_path
    except Exception:
        pass
    return current_path


def browse_input_folder(current_dir):
    """Open a native macOS folder-picker dialog for input folder selection."""
    import subprocess
    try:
        script = 'POSIX path of (choose folder with prompt "Select folder with low-res images:")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            folder = result.stdout.strip().rstrip("/")
            return folder if folder else current_dir
    except Exception:
        pass
    return current_dir


def batch_upscale_folder(input_folder, output_folder, scale_choice, model_path):
    """
    Generator: upscale all images in input_folder and save to output_folder.
    scale_choice: "×2", "×3", or "×4"
    Uses spandrel 4x upscaler; ×2 and ×3 are achieved by resizing down from 4x.
    """
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    if not model_path or not os.path.isfile(model_path):
        yield "⚠️  No upscaler model selected. Set the Upscaler model path above first."
        return

    if not input_folder or not os.path.isdir(input_folder):
        yield "⚠️  Input folder not found. Please pick a valid folder."
        return

    files = sorted([
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])
    if not files:
        yield "⚠️  No image files (.png .jpg .jpeg .webp .bmp .tiff) found in the selected folder."
        return

    out_dir = output_folder.strip() if output_folder and output_folder.strip() else input_folder
    os.makedirs(out_dir, exist_ok=True)

    scale_map = {"×2": 2, "×3": 3, "×4": 4}
    target_scale = scale_map.get(scale_choice, 4)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    log = f"Starting batch upscale — {len(files)} image(s) → {scale_choice}  |  device: {device}\n"
    yield log

    for idx, fname in enumerate(files, 1):
        src_path = os.path.join(input_folder, fname)
        stem, ext = os.path.splitext(fname)
        out_name = f"{stem}_x{target_scale}{ext}"
        dst_path = os.path.join(out_dir, out_name)

        try:
            img = Image.open(src_path).convert("RGB")
            orig_w, orig_h = img.size

            # Run the model at native 4× scale
            upscaled = upscale_image(img, model_path, device)

            # If target is ×2 or ×3, resize down from 4× to the desired scale
            if target_scale != 4:
                new_w = orig_w * target_scale
                new_h = orig_h * target_scale
                upscaled = upscaled.resize((new_w, new_h), Image.LANCZOS)

            # Save, preserving format where possible
            save_ext = ext.lower().lstrip(".")
            fmt = {"jpg": "JPEG", "jpeg": "JPEG"}.get(save_ext, "PNG")
            upscaled.save(dst_path, fmt)

            msg = f"[{idx}/{len(files)}] ✓  {fname}  ({orig_w}×{orig_h} → {upscaled.size[0]}×{upscaled.size[1]})"
        except Exception as e:
            msg = f"[{idx}/{len(files)}] ✗  {fname}  — Error: {e}"

        log += msg + "\n"
        yield log

    log += f"\nDone! {len(files)} image(s) saved to: {out_dir}"
    yield log


def browse_output_folder(current_dir):
    """Open a native macOS folder-picker dialog and return the chosen path."""
    import subprocess
    try:
        script = 'POSIX path of (choose folder with prompt "Select output folder:")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            folder = result.stdout.strip().rstrip("/")
            return folder if folder else current_dir
    except Exception:
        pass
    return current_dir  # user cancelled or error → keep current


def save_image(image, output_dir=None, prompt=""):
    """Save image to output directory."""
    if image is None:
        return "No image to save"
    
    output_dir = get_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = ""
    if prompt:
        prompt_slug = "_" + "".join(c if c.isalnum() else "_" for c in prompt[:30]).strip("_")
    
    filename = f"{timestamp}{prompt_slug}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath, "PNG")
    return f"Saved: {filepath}"


# =============================================================================
# HuggingFace Account Functions
# =============================================================================

def hf_get_status():
    """Return a string describing the current HF login status."""
    try:
        from huggingface_hub import whoami
        info = whoami()
        name = info.get("name") or info.get("fullname") or info.get("username", "unknown")
        return f"✅ Logged in as: **{name}**"
    except Exception:
        return "⚠️ Not logged in"

def hf_login_token(token: str):
    """Log in to HuggingFace with the provided token. Returns a status string."""
    token = (token or "").strip()
    if not token:
        return "⚠️ Please enter a token first."
    try:
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        info = whoami()
        name = info.get("name") or info.get("fullname") or info.get("username", "unknown")
        return f"✅ Logged in as: **{name}**"
    except Exception as e:
        return f"❌ Login failed: {e}"

def hf_logout():
    """Log out from HuggingFace."""
    try:
        from huggingface_hub import logout
        logout()
        return "🔓 Logged out."
    except Exception as e:
        return f"❌ Logout failed: {e}"


# =============================================================================
# Storage Management Functions
# =============================================================================

# Models this app uses (HuggingFace repo IDs)
KNOWN_MODELS = {
    "aydin99/FLUX.2-klein-4B-int8": "FLUX.2-klein-4B (Int8)",
    "black-forest-labs/FLUX.2-klein-4B": "FLUX.2-klein-4B (Base)",
    "black-forest-labs/FLUX.2-klein-9B": "FLUX.2-klein-9B (Base)",
    "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic": "FLUX.2-klein-4B (4bit SDNQ)",
    "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32": "FLUX.2-klein-9B (4bit SDNQ)",
    "Tongyi-MAI/Z-Image-Turbo": "Z-Image Turbo (Full)",
    "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32": "Z-Image Turbo (Quantized)",
    "filipstrand/Z-Image-Turbo-mflux-4bit": "Z-Image Turbo (mflux 4bit)",
    "Lightricks/LTX-Video": "LTX-Video",
}


def get_hf_cache_dir():
    """Get HuggingFace cache directory."""
    return os.environ.get("HF_HUB_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))


def get_dir_size(path):
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total


def format_size(size_bytes):
    """Format bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def scan_downloaded_models():
    """Scan HuggingFace cache for downloaded models used by this app."""
    cache_dir = get_hf_cache_dir()
    models = []
    total_size = 0
    
    if not os.path.exists(cache_dir):
        return [], "0 B"
    
    for repo_id, display_name in KNOWN_MODELS.items():
        # Convert repo_id to cache folder name (owner--model)
        cache_name = f"models--{repo_id.replace('/', '--')}"
        model_path = os.path.join(cache_dir, cache_name)
        
        if os.path.exists(model_path):
            size = get_dir_size(model_path)
            total_size += size
            models.append({
                "repo_id": repo_id,
                "display_name": display_name,
                "cache_name": cache_name,
                "path": model_path,
                "size": size,
                "size_str": format_size(size),
            })
    
    models.sort(key=lambda x: x["size"], reverse=True)
    
    return models, format_size(total_size)


def get_storage_display():
    """Get formatted storage display for Gradio."""
    models, total = scan_downloaded_models()
    
    if not models:
        return "No models downloaded yet. Models will download on first use."
    
    lines = [f"**Total Storage Used: {total}**\n"]
    lines.append("| Model | Size |")
    lines.append("|-------|------|")
    
    for m in models:
        lines.append(f"| {m['display_name']} | {m['size_str']} |")
    
    return "\n".join(lines)


def get_model_choices_for_deletion():
    """Get list of model choices for deletion dropdown."""
    models, _ = scan_downloaded_models()
    choices = []
    for m in models:
        choices.append(f"{m['display_name']} ({m['size_str']})")
    return choices


def delete_model(model_selection):
    """Delete a specific model from cache."""
    global pipe, current_model
    
    if not model_selection:
        return get_storage_display(), get_model_choices_for_deletion(), "No model selected"
    
    models, _ = scan_downloaded_models()
    
    target = None
    for m in models:
        if model_selection.startswith(m['display_name']):
            target = m
            break
    
    if not target:
        return get_storage_display(), get_model_choices_for_deletion(), f"Model not found: {model_selection}"
    
    # Unload pipeline if it's using this model
    model_repo = target['repo_id'].lower()
    if pipe is not None:
        needs_unload = False
        if "klein-4b" in model_repo and current_model and "4b" in current_model.lower():
            needs_unload = True
        elif "klein-9b" in model_repo and current_model and "9b" in current_model.lower():
            needs_unload = True
        elif "z-image" in model_repo.lower() and current_model and "zimage" in current_model.lower():
            needs_unload = True
        
        if needs_unload:
            del pipe
            pipe = None
            current_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    
    try:
        shutil.rmtree(target['path'])
        msg = f"Deleted: {target['display_name']} ({target['size_str']} freed)"
        print(msg)
    except Exception as e:
        msg = f"Error deleting {target['display_name']}: {str(e)}"
        print(msg)
    
    return get_storage_display(), get_model_choices_for_deletion(), msg


def delete_all_models():
    """Delete all downloaded models."""
    global pipe, current_model, current_lora_path
    
    models, total = scan_downloaded_models()
    
    if not models:
        return get_storage_display(), get_model_choices_for_deletion(), "No models to delete"
    
    if pipe is not None:
        del pipe
        pipe = None
        current_model = None
        current_lora_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    deleted = []
    errors = []
    
    for m in models:
        try:
            shutil.rmtree(m['path'])
            deleted.append(m['display_name'])
        except Exception as e:
            errors.append(f"{m['display_name']}: {str(e)}")
    
    if errors:
        msg = f"Deleted {len(deleted)} models. Errors: {'; '.join(errors)}"
    else:
        msg = f"Deleted {len(deleted)} models. {total} freed."
    
    print(msg)
    return get_storage_display(), get_model_choices_for_deletion(), msg


# =============================================================================
# Workflow save / load
# =============================================================================

def list_saved_workflows():
    """Return list of saved workflow folder names, newest first."""
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    entries = []
    for name in sorted(os.listdir(WORKFLOWS_DIR), reverse=True):
        if os.path.isdir(os.path.join(WORKFLOWS_DIR, name)):
            if os.path.exists(os.path.join(WORKFLOWS_DIR, name, "workflow.json")):
                entries.append(name)
    return entries


def save_workflow(wf_name, prompt, height, width, steps, seed, guidance,
                  device, model_choice, model_source, lora_file, lora_strength,
                  img_strength, repeat_count, upscale_enabled, upscale_model_path,
                  num_frames, fps_val, input_images):
    """Serialise all current parameters + reference images to a workflow folder."""
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom   = (wf_name or "").strip().replace(" ", "_")
    slug     = "".join(c if c.isalnum() else "_" for c in (prompt or "")[:30]).strip("_")
    folder_name = f"{timestamp}_{custom}" if custom else (f"{timestamp}_{slug}" if slug else timestamp)
    wf_dir = os.path.join(WORKFLOWS_DIR, folder_name)
    os.makedirs(wf_dir, exist_ok=True)

    # Save reference images
    ref_filenames = []
    if input_images:
        for idx, img_data in enumerate(input_images):
            img = img_data[0] if isinstance(img_data, tuple) else img_data
            if img is not None:
                fname = f"ref_{idx}.png"
                img.save(os.path.join(wf_dir, fname))
                ref_filenames.append(fname)

    data = {
        "name":              wf_name or folder_name,
        "timestamp":         timestamp,
        "prompt":            prompt or "",
        "height":            int(height),
        "width":             int(width),
        "steps":             int(steps),
        "seed":              int(seed),
        "guidance":          float(guidance),
        "device":            device or "",
        "model_choice":      model_choice or MODEL_CHOICES[0],
        "model_source":      model_source or "Local",
        "lora_file":         lora_file or "",
        "lora_strength":     float(lora_strength),
        "img_strength":      float(img_strength),
        "repeat_count":      int(repeat_count or 1),
        "upscale_enabled":   bool(upscale_enabled),
        "upscale_model_path": upscale_model_path or "",
        "num_frames":        int(num_frames or 25),
        "fps":               int(fps_val or 24),
        "reference_images":  ref_filenames,
    }
    with open(os.path.join(wf_dir, "workflow.json"), "w") as f:
        json.dump(data, f, indent=2)

    choices = list_saved_workflows()
    return f"✓ Saved: {folder_name}", gr.update(choices=choices, value=folder_name)


def load_workflow(workflow_name):
    """
    Load a workflow and return values for all UI components.
    Returns 18 values: scalar params × 16 + input_images + status_str.
    """
    _no_op = gr.update()
    if not workflow_name:
        return (_no_op,) * 17 + ("No workflow selected",)
    wf_dir    = os.path.join(WORKFLOWS_DIR, workflow_name)
    json_path = os.path.join(wf_dir, "workflow.json")
    if not os.path.exists(json_path):
        return (_no_op,) * 17 + (f"Not found: {workflow_name}",)

    with open(json_path) as f:
        d = json.load(f)

    # Load reference images back as PIL
    ref_images = []
    for fname in d.get("reference_images", []):
        fpath = os.path.join(wf_dir, fname)
        if os.path.exists(fpath):
            ref_images.append(Image.open(fpath).copy())

    lora_note = ""
    if d.get("lora_file"):
        lora_note = f"  (LoRA was: {os.path.basename(d['lora_file'])})"

    return (
        d.get("prompt", ""),                         # → prompt
        d.get("height", 512),                        # → height
        d.get("width", 512),                         # → width
        d.get("steps", 4),                           # → steps
        d.get("seed", -1),                           # → seed
        d.get("guidance", 3.5),                      # → guidance_scale
        d.get("device", "mps"),                      # → device
        d.get("model_choice", MODEL_CHOICES[0]),     # → model_choice
        d.get("model_source", "Local"),              # → model_source_radio
        d.get("lora_strength", 1.0),                 # → lora_strength
        d.get("img_strength", 0.6),                  # → img_strength
        d.get("repeat_count", 1),                    # → repeat_count
        d.get("upscale_enabled", False),             # → upscale_enabled
        d.get("upscale_model_path", ""),             # → upscale_model_path
        d.get("num_frames", 25),                     # → num_frames
        d.get("fps", 24),                            # → fps_slider
        ref_images if ref_images else None,          # → input_images
        f"✓ Loaded: {workflow_name}{lora_note}",     # → wf_status
    )


def calculate_dimensions_from_ratio(width: int, height: int, target_resolution: str) -> tuple:
    """Calculate output dimensions maintaining aspect ratio for target resolution."""
    if "1536" in target_resolution:
        target_size = 1536
    elif "1280" in target_resolution:
        target_size = 1280
    elif "2048" in target_resolution or "2K" in target_resolution:
        target_size = 2048
    elif "512" in target_resolution:
        target_size = 512
    else:
        target_size = 1024
    
    aspect_ratio = width / height
    
    if aspect_ratio >= 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    
    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))
    
    return new_width, new_height


def on_image_upload(images, current_preset):
    if images is None or len(images) == 0:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value="~1024px")
    
    try:
        first_image = images[0][0] if isinstance(images[0], tuple) else images[0]
        img_width, img_height = first_image.size
    except Exception:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value="~1024px")
    
    preset = current_preset if current_preset in ["~512px", "~1024px", "~1280px", "~1536px (32GB+)"] else "~1024px"
    new_width, new_height = calculate_dimensions_from_ratio(img_width, img_height, preset)
    
    return (
        gr.update(visible=False, value=new_width),
        gr.update(visible=False, value=new_height),
        gr.update(visible=True, value=preset),
    )


def on_resolution_preset_change(preset, images):
    if images is None or len(images) == 0:
        return gr.update(), gr.update()
    
    first_image = images[0][0] if isinstance(images[0], tuple) else images[0]
    img_width, img_height = first_image.size
    new_width, new_height = calculate_dimensions_from_ratio(img_width, img_height, preset)
    
    return gr.update(value=new_width), gr.update(value=new_height)


def update_ui_for_model(model_choice):
    """Update UI visibility and defaults based on model selection."""
    is_flux        = "FLUX" in model_choice
    is_zimage_full = "Full" in model_choice
    is_video       = "LTX-Video" in model_choice
    is_img2img_capable = is_flux or is_zimage_full or is_video

    guidance_default = 3.5 if is_flux else (3.0 if is_video else 0.0)
    steps_default    = 25  if is_video else 4

    return (
        gr.update(visible=is_img2img_capable),  # input_images
        gr.update(visible=False),               # resolution_preset
        gr.update(visible=is_zimage_full),      # lora_file
        gr.update(visible=is_zimage_full),      # lora_strength
        gr.update(visible=is_zimage_full),      # clear_lora_btn
        gr.update(value=guidance_default),      # guidance_scale
        gr.update(visible=is_zimage_full),      # img_strength
        gr.update(visible=is_video),            # video_params_row
        gr.update(visible=not is_video),        # output_image
        gr.update(visible=is_video),            # output_video
        gr.update(value=steps_default),         # steps
    )


# Get available devices at startup
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"

# Create Gradio interface
_css = """
/* ── Reference gallery: images fill the 400px container proportionally ── */
#ref_gallery .grid-wrap          { height: 400px !important; overflow: hidden; }
#ref_gallery .thumbnail-item     { height: 400px !important; }
#ref_gallery .thumbnail-item img { object-fit: contain !important;
                                    height: 100% !important;
                                    max-height: 400px !important; }

/* ── Hide Gradio's built-in elapsed-time badge on output components ── */
.eta-bar, .generating { display: none !important; }


/* ── Iterations number-input spinner arrows → white ── */
#iter_count input[type=number]::-webkit-inner-spin-button,
#iter_count input[type=number]::-webkit-outer-spin-button {
    filter: invert(1);
    opacity: 1;
}
"""

with gr.Blocks(title="Ultra Fast Image Gen", css=_css) as demo:
    gr.Markdown("## Ultra Fast Image Gen")

    with gr.Row():
        # ── Left column: all controls ────────────────────────────────────────
        with gr.Column(scale=1):

            # Model + Device on one row
            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES[0],
                    label="Model",
                    scale=3,
                )
                device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="Device",
                    scale=1,
                )

            # Source + status on one row
            with gr.Row():
                model_source_radio = gr.Radio(
                    choices=["Local", "HF Cache (sync if newer)"],
                    value="Local",
                    label="Source",
                    scale=1,
                )
                model_source_status = gr.Textbox(
                    label="",
                    interactive=False,
                    value="",
                    scale=2,
                    show_label=False,
                    container=False,
                )

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image...",
                lines=2,
            )

            # Reference image input — visible for FLUX + Z-Image Full
            input_images = gr.Gallery(
                label="Reference images (optional) — FLUX: up to 6 · Z-Image Full: 1st only",
                type="pil",
                visible=True,
                columns=4,
                height=400,
                object_fit="contain",
                interactive=True,
                elem_id="ref_gallery",
            )
            resolution_preset = gr.Radio(
                choices=["~512px", "~1024px", "~1280px", "~1536px (32GB+)"],
                value="~1024px",
                label="Output size (longest side, keeps aspect ratio)",
                visible=False,
            )

            # Generation settings collapsed by default
            with gr.Accordion("Generation settings", open=False):
                with gr.Row():
                    steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                    seed = gr.Number(value=-1, label="Seed  (−1 = random)")
                with gr.Row():
                    height = gr.Slider(256, 2048, value=512, step=64, label="Height")
                    width  = gr.Slider(256, 2048, value=512, step=64, label="Width")
                guidance_scale = gr.Slider(
                    0.0, 10.0, value=3.5, step=0.5,
                    label="Guidance scale  (FLUX: 3.5 · Z-Image: 0)",
                )

            # LoRA + image strength — Z-Image Full only
            with gr.Row():
                lora_file = gr.File(
                    label="LoRA  (.safetensors)",
                    file_types=[".safetensors"],
                    file_count="single",
                    type="filepath",
                    visible=False,
                    scale=3,
                )
                clear_lora_btn = gr.Button("✕ LoRA", scale=0, min_width=80, visible=False)
            with gr.Row():
                lora_strength = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.05,
                    label="LoRA strength",
                    visible=False,
                    scale=1,
                )
                img_strength = gr.Slider(
                    0.0, 1.0, value=0.6, step=0.05,
                    label="Image strength  (0 = keep ref · 1 = ignore ref)",
                    visible=False,
                    scale=1,
                )

            # Video parameters — only visible when LTX-Video is selected
            with gr.Row(visible=False) as video_params_row:
                num_frames = gr.Slider(9, 121, value=25, step=8, label="Frames  (9 · 17 · 25 … 121)")
                fps_slider = gr.Slider(8, 30, value=24, step=1, label="FPS")

            with gr.Row(equal_height=True):
                generate_btn  = gr.Button("Generate", variant="primary", size="lg", scale=2)
                with gr.Column(scale=1, min_width=150):
                    with gr.Row(equal_height=True):
                        gr.HTML('<div style="display:flex;align-items:center;justify-content:flex-end;font-weight:bold;white-space:nowrap;padding-right:8px;color:var(--body-text-color,#fff);flex:1">Iterations</div>')
                        repeat_count  = gr.Number(value=1, minimum=1, maximum=100, step=1, show_label=False, precision=0, container=False, min_width=80, elem_id="iter_count")
            seed_info = gr.Textbox(label="Info", interactive=False, lines=1)

            with gr.Accordion("Workflows", open=False):
                wf_name_input = gr.Textbox(
                    label="Name (optional)",
                    placeholder="My Portrait Style",
                )
                with gr.Row():
                    save_wf_btn = gr.Button("Save workflow", scale=1)
                    wf_save_status = gr.Textbox(
                        label="", interactive=False,
                        show_label=False, container=False, scale=2,
                    )
                with gr.Row():
                    wf_dropdown = gr.Dropdown(
                        choices=list_saved_workflows(),
                        label="Load saved workflow",
                        scale=3,
                    )
                    load_wf_btn  = gr.Button("Load", variant="primary", scale=1)
                    refresh_wf_btn = gr.Button("↺", scale=0, min_width=40)
                wf_load_status = gr.Textbox(label="", interactive=False, show_label=False, container=False)

        # ── Right column: output ─────────────────────────────────────────────
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output", type="pil", interactive=False)
            output_video = gr.Video(label="Output Video", visible=False)
            with gr.Row():
                save_btn        = gr.Button("Save image", scale=2)
                open_folder_btn = gr.Button("Open folder", scale=1)
                auto_save       = gr.Checkbox(label="Auto-save", value=True, scale=0)
            save_status = gr.Textbox(
                label="", interactive=False, lines=1,
                show_label=False, container=False,
            )
            with gr.Row():
                output_dir = gr.Textbox(
                    label="Output folder", value=DEFAULT_OUTPUT_DIR, scale=4,
                )
                browse_btn = gr.Button("📂", scale=0, min_width=48)

            with gr.Row():
                upscale_enabled = gr.Checkbox(label="4× Upscale", value=False, scale=0, min_width=100)
                upscale_model_path = gr.Textbox(
                    label="Upscaler model path  (.pth or .safetensors)",
                    placeholder="e.g. /path/to/4x-UltraSharp.pth",
                    scale=4,
                )
                browse_upscaler_btn = gr.Button("📂", scale=0, min_width=48)

            with gr.Accordion("Batch Upscaler", open=False):
                with gr.Row():
                    batch_input_folder = gr.Textbox(
                        label="Input folder (low-res images)",
                        placeholder="/path/to/lowres/",
                        scale=4,
                    )
                    browse_batch_input_btn = gr.Button("📂", scale=0, min_width=48)
                with gr.Row():
                    batch_output_folder = gr.Textbox(
                        label="Output folder (leave blank = save next to originals)",
                        placeholder="/path/to/output/  (optional)",
                        scale=4,
                    )
                    browse_batch_output_btn = gr.Button("📂", scale=0, min_width=48)
                with gr.Row():
                    batch_scale_radio = gr.Radio(
                        choices=["×2", "×3", "×4"],
                        value="×4",
                        label="Scale factor",
                        scale=1,
                    )
                    run_batch_upscale_btn = gr.Button("Run Batch Upscale", variant="primary", scale=2)
                batch_upscale_log = gr.Textbox(
                    label="Progress",
                    interactive=False,
                    lines=6,
                    max_lines=20,
                )

            with gr.Accordion("Storage & Updates", open=False):
                with gr.Tabs():
                    with gr.Tab("Local"):
                        storage_display = gr.Markdown(value=get_storage_display())
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=get_model_choices_for_deletion(),
                                label="Select model to delete",
                                scale=3,
                            )
                            delete_btn = gr.Button("Delete", variant="secondary", scale=1)
                        with gr.Row():
                            refresh_btn    = gr.Button("Refresh", scale=1)
                            delete_all_btn = gr.Button("Delete ALL", variant="stop", scale=1)
                        storage_status = gr.Textbox(label="Status", interactive=False, lines=1)

                    with gr.Tab("HF Sync"):
                        versions_display = gr.Markdown(value=get_versions_display())
                        with gr.Row():
                            check_versions_btn = gr.Button("Refresh", scale=1)
                            sync_all_btn       = gr.Button("Sync newer → local", variant="primary", scale=1)
                        sync_status = gr.Textbox(label="Sync status", interactive=False, lines=1)

                    with gr.Tab("Online"):
                        online_versions_display = gr.Markdown(value="_Click **Check** to see results._")
                        with gr.Row():
                            check_online_btn        = gr.Button("Check", scale=1)
                            download_all_online_btn = gr.Button("Download all", variant="secondary", scale=1)
                        with gr.Row():
                            update_model_dropdown  = gr.Dropdown(
                                choices=[], label="Select to download / update", scale=3,
                            )
                            download_selected_btn  = gr.Button("Download", variant="primary", scale=1)
                        online_update_status = gr.Textbox(label="Status", interactive=False, lines=1)

                    with gr.Tab("HF Account"):
                        hf_status_display = gr.Markdown(value=hf_get_status())
                        with gr.Row():
                            hf_token_input = gr.Textbox(
                                label="HuggingFace token",
                                placeholder="hf_...",
                                type="password",
                                scale=4,
                            )
                            hf_login_btn  = gr.Button("Login",  variant="primary",   scale=1)
                            hf_logout_btn = gr.Button("Logout", variant="secondary",  scale=1)
                        gr.Markdown(
                            "_Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
                            "(needs **Read** permission). Required for gated models like FLUX.2-klein-9B._"
                        )
                        hf_account_status = gr.Textbox(label="Result", interactive=False, lines=1)

    # ── Examples ─────────────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            ["A majestic mountain landscape at sunset, dramatic lighting, cinematic"],
            ["Portrait of a young woman, soft studio lighting, professional photography"],
            ["Cyberpunk city street at night, neon lights, rain reflections"],
            ["A cute cat wearing a tiny hat, studio photo, soft lighting"],
            ["Abstract art, vibrant colors, fluid shapes, modern design"],
        ],
        inputs=[prompt],
    )

    # ── Event handlers ───────────────────────────────────────────────────────
    _ui_model_outputs = [
        input_images, resolution_preset,
        lora_file, lora_strength, clear_lora_btn,
        guidance_scale, img_strength,
        video_params_row, output_image, output_video, steps,
    ]

    model_choice.change(
        fn=update_ui_for_model,
        inputs=[model_choice],
        outputs=_ui_model_outputs,
    )
    model_choice.change(
        fn=check_model_source_status,
        inputs=[model_choice, model_source_radio],
        outputs=[model_source_status],
    )
    model_source_radio.change(
        fn=check_model_source_status,
        inputs=[model_choice, model_source_radio],
        outputs=[model_source_status],
    )

    input_images.change(
        fn=on_image_upload,
        inputs=[input_images, resolution_preset],
        outputs=[width, height, resolution_preset],
    )
    resolution_preset.change(
        fn=on_resolution_preset_change,
        inputs=[resolution_preset, input_images],
        outputs=[width, height],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt, height, width, steps, seed, guidance_scale, device,
            model_choice, model_source_radio, input_images,
            lora_file, lora_strength, img_strength,
            repeat_count, auto_save, output_dir,
            upscale_enabled, upscale_model_path,
            num_frames, fps_slider,
        ],
        outputs=[output_image, output_video, seed_info],
    )

    def manual_save(image, out_dir, prompt_text):
        if image is None:
            return "No image to save"
        return save_image(image, out_dir, prompt_text)

    save_btn.click(
        fn=manual_save,
        inputs=[output_image, output_dir, prompt],
        outputs=[save_status],
    )

    def open_output_folder(out_dir):
        import subprocess
        folder = get_output_dir(out_dir)
        subprocess.run(["open", folder])
        return f"Opened: {folder}"

    open_folder_btn.click(
        fn=open_output_folder,
        inputs=[output_dir],
        outputs=[save_status],
    )

    browse_btn.click(
        fn=browse_output_folder,
        inputs=[output_dir],
        outputs=[output_dir],
        show_progress="hidden",
    )

    browse_upscaler_btn.click(
        fn=browse_upscaler_file,
        inputs=[upscale_model_path],
        outputs=[upscale_model_path],
        show_progress="hidden",
    )

    browse_batch_input_btn.click(
        fn=browse_input_folder,
        inputs=[batch_input_folder],
        outputs=[batch_input_folder],
        show_progress="hidden",
    )

    browse_batch_output_btn.click(
        fn=browse_output_folder,
        inputs=[batch_output_folder],
        outputs=[batch_output_folder],
        show_progress="hidden",
    )

    run_batch_upscale_btn.click(
        fn=batch_upscale_folder,
        inputs=[batch_input_folder, batch_output_folder, batch_scale_radio, upscale_model_path],
        outputs=[batch_upscale_log],
    )

    clear_lora_btn.click(fn=clear_lora, outputs=[lora_file, seed_info])
    lora_strength.change(fn=update_lora_strength, inputs=[lora_strength], outputs=[seed_info])

    def refresh_storage():
        new_choices = get_model_choices_for_deletion()
        return get_storage_display(), gr.update(choices=new_choices, value=None), "", get_versions_display()

    refresh_btn.click(
        fn=refresh_storage,
        outputs=[storage_display, model_dropdown, storage_status, versions_display],
    )
    delete_btn.click(
        fn=delete_model,
        inputs=[model_dropdown],
        outputs=[storage_display, model_dropdown, storage_status],
    )
    delete_all_btn.click(
        fn=delete_all_models,
        outputs=[storage_display, model_dropdown, storage_status],
    )
    check_versions_btn.click(fn=get_versions_display, outputs=[versions_display])

    def do_sync_all():
        msg = sync_all_newer_from_hf()
        new_choices = get_model_choices_for_deletion()
        return get_versions_display(), get_storage_display(), gr.update(choices=new_choices, value=None), msg

    sync_all_btn.click(
        fn=do_sync_all,
        outputs=[versions_display, storage_display, model_dropdown, sync_status],
    )
    check_online_btn.click(
        fn=check_online_versions,
        outputs=[online_versions_display, update_model_dropdown],
    )
    download_selected_btn.click(
        fn=download_model_update,
        inputs=[update_model_dropdown],
        outputs=[online_update_status, online_versions_display, update_model_dropdown],
    )
    download_all_online_btn.click(
        fn=download_all_online_updates,
        outputs=[online_update_status, online_versions_display, update_model_dropdown],
    )

    # ── HF Account handlers ───────────────────────────────────────────────────
    hf_login_btn.click(
        fn=hf_login_token,
        inputs=[hf_token_input],
        outputs=[hf_account_status],
    ).then(
        fn=hf_get_status,
        outputs=[hf_status_display],
    )
    hf_logout_btn.click(
        fn=hf_logout,
        outputs=[hf_account_status],
    ).then(
        fn=hf_get_status,
        outputs=[hf_status_display],
    )

    # ── Workflow handlers ─────────────────────────────────────────────────────
    _wf_load_outputs = [
        prompt, height, width, steps, seed, guidance_scale, device,
        model_choice, model_source_radio,
        lora_strength, img_strength, repeat_count,
        upscale_enabled, upscale_model_path,
        num_frames, fps_slider,
        input_images,
        wf_load_status,
    ]

    save_wf_btn.click(
        fn=save_workflow,
        inputs=[
            wf_name_input, prompt, height, width, steps, seed, guidance_scale,
            device, model_choice, model_source_radio,
            lora_file, lora_strength, img_strength, repeat_count,
            upscale_enabled, upscale_model_path, num_frames, fps_slider,
            input_images,
        ],
        outputs=[wf_save_status, wf_dropdown],
    )

    load_wf_btn.click(
        fn=load_workflow,
        inputs=[wf_dropdown],
        outputs=_wf_load_outputs,
    )

    refresh_wf_btn.click(
        fn=lambda: gr.update(choices=list_saved_workflows(), value=None),
        outputs=[wf_dropdown],
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
