"""
Z-Image Turbo - Gradio Web Interface

Fast image generation on Apple Silicon.
Uses quantized model for speed, full model when LoRA is loaded.
"""

import os
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import torch
import sdnq  # Required for quantized model
import gradio as gr
from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler

# Global pipeline, device, and model state
pipe = None
current_device = None
current_lora_path = None
current_model_type = None  # "quantized" or "full"


def get_available_devices():
    """Get list of available devices."""
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def load_pipeline(device="mps", use_full_model=False):
    """Load the pipeline (cached globally). Switches between quantized and full model."""
    global pipe, current_device, current_model_type, current_lora_path

    model_type = "full" if use_full_model else "quantized"

    # Return cached pipeline if same device and model type
    if pipe is not None and current_device == device and current_model_type == model_type:
        return pipe

    # Need to reload - clear existing pipeline
    if pipe is not None:
        print(f"Switching from {current_model_type} to {model_type} model...")
        del pipe
        current_lora_path = None  # Reset LoRA state when switching models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    # Use Euler with beta sigmas for cleaner images
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

    current_device = device
    current_model_type = model_type
    print(f"Pipeline loaded on {device}! (Model: {model_type})")
    return pipe


def load_lora(lora_file, lora_strength: float, device: str):
    """Load or update LoRA adapter using native diffusers support."""
    global current_lora_path, pipe

    # Handle no file selected
    if lora_file is None or lora_file == "":
        if current_lora_path is not None:
            print("Unloading current LoRA...")
            pipe.unload_lora_weights()
            current_lora_path = None
        return "No LoRA loaded"

    # Get the path from the file upload
    lora_path = lora_file if isinstance(lora_file, str) else lora_file.name

    if not os.path.exists(lora_path):
        return f"LoRA file not found: {lora_path}"

    if not lora_path.endswith('.safetensors'):
        return "Please select a .safetensors file"

    # If same LoRA, just update scale
    if current_lora_path == lora_path:
        pipe.set_adapters(["default"], adapter_weights=[lora_strength])
        return f"Updated LoRA strength to {lora_strength}"

    # Unload old LoRA if exists
    if current_lora_path is not None:
        print(f"Unloading previous LoRA: {current_lora_path}")
        pipe.unload_lora_weights()

    # Load new LoRA using native diffusers support
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


def generate_image(prompt, height, width, steps, seed, guidance, device, lora_file, lora_strength):
    """Generate an image from the prompt."""
    global pipe

    # Determine if we need full model (LoRA selected) or quantized (no LoRA)
    use_lora = lora_file is not None and lora_file != ""

    # Load appropriate pipeline
    pipe = load_pipeline(device, use_full_model=use_lora)

    # Handle LoRA loading/updating (only if using full model)
    if use_lora:
        lora_status = load_lora(lora_file, lora_strength, device)

    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    # Use appropriate generator for device
    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(int(seed))
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(int(seed))

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
        ).images[0]

    lora_name = os.path.basename(lora_file) if lora_file else None
    lora_info = f" | LoRA: {lora_name} ({lora_strength})" if lora_name else ""
    model_info = "full" if use_lora else "quant"
    cfg_info = f" | CFG: {guidance}" if guidance > 0 else ""
    return image, f"Seed: {seed} | Model: {model_info} | Device: {device}{cfg_info}{lora_info}"


def clear_lora():
    """Clear the current LoRA."""
    global current_lora_path, pipe
    if current_lora_path is not None and pipe is not None:
        pipe.unload_lora_weights()
        current_lora_path = None
    return None, "LoRA cleared - will use quantized model next generation"


# Get available devices at startup
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"

# Create Gradio interface
# delete_cache=(60, 60) means check every 60 seconds, delete files older than 60 seconds
with gr.Blocks(title="Z-Image Turbo", delete_cache=(60, 60)) as demo:
    gr.Markdown("""
    # Z-Image Turbo

    Fast image generation on Apple Silicon with LoRA support.

    **Quantized model (3.5GB):** Used when no LoRA is loaded - fast!
    **Full model (24GB):** Used when LoRA is loaded - slower but supports adapters
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )

            with gr.Row():
                height = gr.Slider(256, 1024, value=768, step=64, label="Height")
                width = gr.Slider(256, 1024, value=768, step=64, label="Width")

            with gr.Row():
                steps = gr.Slider(1, 50, value=7, step=1, label="Steps")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")

            with gr.Row():
                guidance_scale = gr.Slider(
                    0.0, 10.0, value=1.0, step=0.5,
                    label="Guidance Scale (CFG)",
                    info="1.0=Z-Image Turbo default, 0=no guidance"
                )

            with gr.Row():
                device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="Device",
                    info="MPS=Mac, CUDA=NVIDIA (experimental), CPU=slow"
                )

            # LoRA section
            gr.Markdown("### LoRA Settings (loads full model when used)")

            with gr.Row():
                lora_file = gr.File(
                    label="LoRA File",
                    file_types=[".safetensors"],
                    file_count="single",
                    type="filepath",
                )
                clear_lora_btn = gr.Button("Clear LoRA", scale=0, min_width=100)

            lora_strength = gr.Slider(
                0.0, 2.0, value=1.0, step=0.05,
                label="LoRA Strength",
                info="1.0 = full effect, 0.5 = half effect"
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
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, steps, seed, guidance_scale, device, lora_file, lora_strength],
        outputs=[output_image, seed_info],
    )

    clear_lora_btn.click(
        fn=clear_lora,
        outputs=[lora_file, seed_info],
    )

    # Live update LoRA strength
    lora_strength.change(
        fn=update_lora_strength,
        inputs=[lora_strength],
        outputs=[seed_info],
    )


if __name__ == "__main__":
    demo.launch()
