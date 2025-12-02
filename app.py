"""
Z-Image Turbo UINT4 - Gradio Web Interface

Fast image generation on Apple Silicon using the quantized uint4 model.
Now with LoRA support!
"""

import os
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import torch
import sdnq
import gradio as gr
from diffusers import ZImagePipeline

from lora_zimage import load_lora_for_pipeline, LoRANetwork

# Global pipeline, device, and LoRA state
pipe = None
current_device = None
current_lora: LoRANetwork = None
current_lora_path = None


def get_available_devices():
    """Get list of available devices."""
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def load_pipeline(device="mps"):
    """Load the pipeline (cached globally)."""
    global pipe, current_device

    # Reload if device changed
    if pipe is not None and current_device == device:
        return pipe

    if pipe is not None:
        print(f"Switching device from {current_device} to {device}...")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Loading Z-Image-Turbo UINT4 on {device}...")

    # Use float16 for CUDA, float32 for MPS/CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = ZImagePipeline.from_pretrained(
        "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    pipe.to(device)
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    current_device = device
    print(f"Pipeline loaded on {device}!")
    return pipe


def load_lora(lora_file, lora_strength: float, device: str):
    """Load or update LoRA adapter."""
    global current_lora, current_lora_path, pipe

    # Handle no file selected
    if lora_file is None or lora_file == "":
        if current_lora is not None:
            print("Removing current LoRA...")
            current_lora.remove()
            current_lora = None
            current_lora_path = None
        return "No LoRA loaded"

    # Get the path from the file upload
    lora_path = lora_file if isinstance(lora_file, str) else lora_file.name

    if not os.path.exists(lora_path):
        return f"LoRA file not found: {lora_path}"

    if not lora_path.endswith('.safetensors'):
        return "Please select a .safetensors file"

    # Make sure pipeline is loaded
    pipe = load_pipeline(device)

    # If same LoRA, just update multiplier
    if current_lora is not None and current_lora_path == lora_path:
        current_lora.multiplier = lora_strength
        return f"Updated LoRA strength to {lora_strength}"

    # Remove old LoRA if exists
    if current_lora is not None:
        print(f"Removing previous LoRA: {current_lora_path}")
        current_lora.remove()
        current_lora = None

    # Load new LoRA
    try:
        lora_name = os.path.basename(lora_path)
        print(f"Loading LoRA: {lora_path}")
        current_lora = load_lora_for_pipeline(
            pipe,
            lora_path,
            multiplier=lora_strength,
            device=device,
            dtype=torch.float32,  # float32 for MPS compatibility
        )
        current_lora_path = lora_path
        return f"Loaded LoRA: {lora_name} (strength={lora_strength})"
    except Exception as e:
        current_lora = None
        current_lora_path = None
        return f"Error loading LoRA: {str(e)}"


def update_lora_strength(strength: float):
    """Update the LoRA multiplier without reloading."""
    global current_lora
    if current_lora is not None:
        current_lora.multiplier = strength
        return f"LoRA strength updated to {strength}"
    return "No LoRA loaded"


def generate_image(prompt, height, width, steps, seed, device, lora_file, lora_strength):
    """Generate an image from the prompt."""
    global current_lora

    pipe = load_pipeline(device)

    # Handle LoRA loading/updating
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
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

    lora_name = os.path.basename(lora_file) if lora_file else None
    lora_info = f" | LoRA: {lora_name} ({lora_strength})" if lora_name else ""
    return image, f"Seed: {seed} | Device: {device}{lora_info}"


def clear_lora():
    """Clear the current LoRA."""
    global current_lora, current_lora_path
    if current_lora is not None:
        current_lora.remove()
        current_lora = None
        current_lora_path = None
    return None, "LoRA cleared"


# Get available devices at startup
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"

# Create Gradio interface
with gr.Blocks(title="Z-Image Turbo UINT4") as demo:
    gr.Markdown("""
    # Z-Image Turbo UINT4

    Fast image generation using the quantized 3.5GB model with LoRA support.

    **Model:** [Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32)
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
                steps = gr.Slider(1, 10, value=7, step=1, label="Steps")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")

            with gr.Row():
                device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="Device",
                    info="MPS=Mac, CUDA=NVIDIA (experimental), CPU=slow"
                )

            # LoRA section
            gr.Markdown("### LoRA Settings")

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
        inputs=[prompt, height, width, steps, seed, device, lora_file, lora_strength],
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
    # delete_cache=(0, 0) means delete temp files immediately after they're no longer needed
    # This prevents generated images from being stored on disk longer than necessary
    demo.launch(delete_cache=(0, 0))
