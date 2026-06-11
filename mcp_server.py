"""
Ultra Fast Image Gen - MCP Server

Exposes image generation capabilities to MCP clients (e.g., opencode, Claude Desktop).
Allows AI agents to generate images and save them directly to project directories.
"""

import os
import sys
import subprocess
from pathlib import Path

def _check_dependencies():
    """Ensure required Python packages are installed before starting."""
    required = ["sdnq", "diffusers", "transformers", "accelerate", "pillow"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print("🔧 Auto-installing requirements...", file=sys.stderr)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        req_file = os.path.join(script_dir, "requirements.txt")
        if os.path.exists(req_file):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("✅ Dependencies installed successfully.", file=sys.stderr)
        else:
            print(f"❌ requirements.txt not found at {req_file}", file=sys.stderr)
            sys.exit(1)

_check_dependencies()

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Ultra Fast Image Gen")


@mcp.tool()
def generate_image(
    prompt: str,
    output_path: str,
    model: str = "zimage-quant",
    width: int = 512,
    height: int = 512,
    steps: int = 5,
    guidance: float = 3.5,
) -> str:
    """
    Generates a new AI image from text and saves it to the specified path.
    Use for creating website banners, icons, hero images, or UI elements from scratch.
    
    Args:
        prompt: Detailed text description (style, lighting, composition, aspect ratio).
        output_path: Target file path (e.g., 'public/images/banner.png', 'src/assets/hero.jpg').
        model: 'zimage-quant' (ultra-fast, lowest memory, default), 'flux2-4b-sdnq' (high quality), 
               'flux2-9b-sdnq' (highest quality), 'flux2-4b-int8', 'zimage-full' (LoRA), or 'anima'.
               Default is zimage-quant for speed. Ask for flux2-4b-sdnq or flux2-9b-sdnq for higher quality.
        width: Image width in pixels (e.g., 1024 for banners, 512 for icons).
        height: Image height in pixels (e.g., 512 for banners, 512 for icons).
        steps: Inference steps (5 for zimage, 28 for flux/anima).
        guidance: Classifier-free guidance scale (default 3.5, ignored by zimage).
    
    Returns:
        Success message with saved path, or error details.
    """
    abs_output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(abs_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable, 
        os.path.join(script_dir, "generate.py"), 
        model, 
        prompt, 
        "--width", str(width), 
        "--height", str(height), 
        "--output", abs_output_path
    ]
    
    if model.startswith("zimage") or model == "anima":
        cmd.extend(["--steps", str(steps)])
    else:
        cmd.extend(["--steps", str(steps if steps > 5 else 28), "--guidance", str(guidance)])
        
    try:
        # Add a 2-hour timeout to prevent indefinite hangs if the subprocess stalls on exit
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            cwd=script_dir,
            timeout=7200
        )
        return f"Successfully generated image and saved to `{abs_output_path}`.\nDetails: {result.stdout.strip()}"
    except subprocess.TimeoutExpired:
        return f"Failed to generate image: Process timed out after 2 hours."
    except subprocess.CalledProcessError as e:
        return f"Failed to generate image. Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Unexpected error during image generation: {str(e)}"


@mcp.tool()
def edit_image(
    prompt: str,
    input_image_paths: list[str],
    output_path: str,
    model: str = "flux2-4b-sdnq",
    width: int = 512,
    height: int = 512,
    steps: int = 28,
    guidance: float = 3.5,
) -> str:
    """
    Edits or transforms existing images based on a text prompt (image-to-image).
    Use when asked to modify, restyle, or transform existing assets (e.g., "change the background to dark", "make the logo 3D").
    Supported models: 'flux2-4b-sdnq', 'flux2-9b-sdnq', 'flux2-4b-int8'. (Does not work with zimage-quant).
    
    Args:
        prompt: Text description of the desired changes or final result.
        input_image_paths: List of 1 to 6 absolute or relative paths to existing reference images.
        output_path: Target file path for the edited image.
        model: 'flux2-4b-sdnq' (recommended), 'flux2-9b-sdnq' (highest quality), or 'flux2-4b-int8'.
        width: Output image width in pixels.
        height: Output image height in pixels.
        steps: Inference steps (default 28).
        guidance: Classifier-free guidance scale (default 3.5).
    
    Returns:
        Success message with saved path, or error details.
    """
    abs_output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(abs_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    abs_input_paths = [os.path.abspath(p) for p in input_image_paths]
    for p in abs_input_paths:
        if not os.path.exists(p):
            return f"Failed to edit image: Input image not found at `{p}`"

    cmd = [
        sys.executable, 
        os.path.join(script_dir, "generate.py"), 
        model, 
        prompt, 
        "--width", str(width), 
        "--height", str(height), 
        "--output", abs_output_path,
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--input-images"
    ] + abs_input_paths
        
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            cwd=script_dir,
            timeout=7200
        )
        return f"Successfully edited image and saved to `{abs_output_path}`.\nDetails: {result.stdout.strip()}"
    except subprocess.TimeoutExpired:
        return f"Failed to edit image: Process timed out after 2 hours."
    except subprocess.CalledProcessError as e:
        return f"Failed to edit image. Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Unexpected error during image editing: {str(e)}"


if __name__ == "__main__":
    mcp.run()
