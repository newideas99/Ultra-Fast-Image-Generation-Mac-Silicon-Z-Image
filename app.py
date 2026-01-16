"""
Flux Image Generator - Gradio Web Interface

Fast image generation on Apple Silicon and CUDA.
Supports multiple models:
- Z-Image Turbo (quantized/full)
- FLUX.2-klein-4B (int8 quantized)

FLUX.2-klein also supports image-to-image editing!
"""

import os
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import torch
import gradio as gr
from PIL import Image
import json
import atexit
import shutil
import tempfile


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
current_device = None
current_model = None  # "zimage-quant", "zimage-full", "flux2-klein-int8"
current_lora_path = None

# Model choices
MODEL_CHOICES = [
    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)",
    "FLUX.2-klein-4B (Int8)",
    "Z-Image Turbo (Quantized - Fast)",
    "Z-Image Turbo (Full - LoRA support)",
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


def load_zimage_pipeline(device="mps", use_full_model=False):
    """Load Z-Image pipeline (quantized or full)."""
    import sdnq  # Required for quantized model
    from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler
    
    if use_full_model:
        print(f"Loading Z-Image-Turbo (full precision) on {device}...")
        dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        print(f"Loading Z-Image-Turbo UINT4 (quantized) on {device}...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
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
    
    print(f"Loading FLUX.2-klein-4B (int8 quantized) on {device}...")
    print_memory("Before loading")
    
    model_path = snapshot_download("aydin99/FLUX.2-klein-4B-int8")
    
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
    
    print(f"Loading FLUX.2-klein-4B (4bit SDNQ) on {device}...")
    print_memory("Before loading")
    
    print("  Loading tokenizer from base model (SDNQ model missing vocab files)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        subfolder="tokenizer",
        use_fast=False,
    )
    
    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
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


def load_pipeline(model_choice: str, device: str = "mps"):
    global pipe, current_device, current_model, current_lora_path
    
    if "Quantized" in model_choice:
        model_type = "zimage-quant"
    elif "Full" in model_choice:
        model_type = "zimage-full"
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


def generate_image(
    prompt, 
    height, 
    width, 
    steps, 
    seed, 
    guidance, 
    device, 
    model_choice,
    input_images,
    lora_file, 
    lora_strength
):
    global pipe
    
    if "Z-Image" in model_choice and lora_file is not None and lora_file != "":
        model_choice = "Z-Image Turbo (Full - LoRA support)"
    
    pipe = load_pipeline(model_choice, device)
    
    if current_model == "zimage-full" and lora_file:
        load_lora(lora_file, lora_strength, device)
    
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()
    
    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(int(seed))
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(int(seed))
    
    print_memory("Before generation")
    
    with torch.inference_mode():
        if current_model in ("flux2-klein-int8", "flux2-klein-sdnq"):
            images_to_process = None
            if input_images is not None and len(input_images) > 0:
                img_w, img_h = int(width), int(height)
                images_to_process = []
                for img_data in input_images[:6]:
                    img = img_data[0] if isinstance(img_data, tuple) else img_data
                    resized = img.copy().resize((img_w, img_h), Image.LANCZOS)
                    if resized.mode != "RGB":
                        resized = resized.convert("RGB")
                    images_to_process.append(resized)
                print_memory(f"After resizing {len(images_to_process)} image(s)")
                
                if hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
                    pipe.vae.disable_tiling()
                
                image = pipe(
                    prompt=prompt,
                    image=images_to_process if len(images_to_process) > 1 else images_to_process[0],
                    height=img_h,
                    width=img_w,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    generator=generator,
                ).images[0]
                
                if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                
                mode = f"img2img ({len(images_to_process)} ref)"
            else:
                image = pipe(
                    prompt=prompt,
                    height=int(height),
                    width=int(width),
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    generator=generator,
                ).images[0]
                mode = "txt2img"
        else:
            image = pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
            ).images[0]
            mode = "txt2img"
    
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
    
    lora_name = os.path.basename(lora_file) if lora_file else None
    lora_info = f" | LoRA: {lora_name} ({lora_strength})" if lora_name else ""
    cfg_info = f" | CFG: {guidance}" if guidance > 0 else ""
    
    model_short = {
        "zimage-quant": "Z-Image (quant)",
        "zimage-full": "Z-Image (full)",
        "flux2-klein-int8": "FLUX.2-klein (int8)",
        "flux2-klein-sdnq": "FLUX.2-klein (4bit)",
    }.get(current_model, current_model)
    
    return image, f"Seed: {seed} | Model: {model_short} | Mode: {mode} | Device: {device}{cfg_info}{lora_info}"


def clear_lora():
    """Clear the current LoRA."""
    global current_lora_path, pipe
    if current_lora_path is not None and pipe is not None:
        pipe.unload_lora_weights()
        current_lora_path = None
    return None, "LoRA cleared"


def calculate_dimensions_from_ratio(width: int, height: int, target_resolution: str) -> tuple:
    """Calculate output dimensions maintaining aspect ratio for target resolution."""
    if "1536" in target_resolution:
        target_size = 1536
    elif "1280" in target_resolution:
        target_size = 1280
    elif "2048" in target_resolution or "2K" in target_resolution:
        target_size = 2048
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
    
    preset = current_preset if current_preset in ["~1024px", "~1280px", "~1536px (32GB+)"] else "~1024px"
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
    is_flux = "FLUX" in model_choice
    is_zimage_full = "Full" in model_choice
    
    guidance_default = 3.5 if is_flux else 0.0
    
    return (
        gr.update(visible=is_flux),  # img2img_label
        gr.update(visible=is_flux),  # input_image
        gr.update(visible=is_flux),  # resolution_preset
        gr.update(visible=is_zimage_full),  # lora_label
        gr.update(visible=is_zimage_full),  # lora_file
        gr.update(visible=is_zimage_full),  # lora_strength
        gr.update(visible=is_zimage_full),  # clear_lora_btn
        gr.update(value=guidance_default),  # guidance_scale
    )


# Get available devices at startup
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"

# Create Gradio interface
with gr.Blocks(title="Ultra Fast Image Gen") as demo:
    gr.Markdown("""
    # Ultra Fast Image Gen
    
    AI image generation and editing on Apple Silicon and CUDA.
    
    **Models:**
    - **FLUX.2-klein-4B (Int8):** 8GB, supports image-to-image editing (default)
    - **Z-Image Turbo (Quantized):** 3.5GB, fastest, no LoRA
    - **Z-Image Turbo (Full):** 24GB, slower, LoRA support
    
    **Resolutions:** Up to 2048px for txt2img. Image-to-image: 1K (16GB) or 1.5K (32GB+).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Model selection
            model_choice = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES[0],
                label="Model",
                info="FLUX.2-klein supports image editing"
            )
            
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )
            
            # Image input (FLUX only) - visible by default since FLUX is default
            img2img_label = gr.Markdown("### Image Input (FLUX.2-klein only - up to 6 images)", visible=True)
            input_images = gr.Gallery(
                label="Input Images (optional - for image-to-image)",
                type="pil",
                visible=True,
                columns=3,
                height="auto",
                interactive=True,
            )

            resolution_preset = gr.Radio(
                choices=["~1024px", "~1280px", "~1536px (32GB+)"],
                value="~1024px",
                label="Output Resolution (longest side)",
                info="Maintains your image's aspect ratio",
                visible=False,
            )
            
            with gr.Row():
                height = gr.Slider(256, 2048, value=512, step=64, label="Height")
                width = gr.Slider(256, 2048, value=512, step=64, label="Width")

            with gr.Row():
                steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")

            with gr.Row():
                guidance_scale = gr.Slider(
                    0.0, 10.0, value=3.5, step=0.5,
                    label="Guidance Scale (CFG)",
                    info="FLUX: 3.5 recommended, Z-Image: 0"
                )

            with gr.Row():
                device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="Device",
                    info="MPS=Mac, CUDA=NVIDIA, CPU=slow"
                )

            # LoRA section (Z-Image Full only) - no Group wrapper for visibility to work
            lora_label = gr.Markdown("### LoRA Settings (Z-Image Full only)", visible=False)
            with gr.Row():
                lora_file = gr.File(
                    label="LoRA File",
                    file_types=[".safetensors"],
                    file_count="single",
                    type="filepath",
                    visible=False,
                )
                clear_lora_btn = gr.Button("Clear LoRA", scale=0, min_width=100, visible=False)

            lora_strength = gr.Slider(
                0.0, 2.0, value=1.0, step=0.05,
                label="LoRA Strength",
                info="1.0 = full effect, 0.5 = half effect",
                visible=False,
            )

            generate_btn = gr.Button("Generate", variant="primary")
            seed_info = gr.Textbox(label="Generation Info", interactive=False)

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")

    # Examples
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

    # Event handlers
    model_choice.change(
        fn=update_ui_for_model,
        inputs=[model_choice],
        outputs=[img2img_label, input_images, resolution_preset, lora_label, lora_file, lora_strength, clear_lora_btn, guidance_scale],
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
            model_choice, input_images, lora_file, lora_strength
        ],
        outputs=[output_image, seed_info],
    )

    clear_lora_btn.click(
        fn=clear_lora,
        outputs=[lora_file, seed_info],
    )

    lora_strength.change(
        fn=update_lora_strength,
        inputs=[lora_strength],
        outputs=[seed_info],
    )


if __name__ == "__main__":
    demo.launch()
