# Ultra Fast Image Gen MCP (Image Generation & Editing)

Use this skill when the user asks to create, generate, edit, transform, or add visual assets (images, banners, icons, backgrounds) to a website project.

## RAM & Resource Awareness (Critical)

**Before generating images, you MUST check available system RAM.** Image generation is memory-intensive and can crash the system or OOM the process. Be strict — never torch the user's RAM even if they insist on batch generation.

### Model Memory Requirements (Approximate)
| Model | Min RAM | Recommended | Max Resolution |
|-------|---------|-------------|----------------|
| `zimage-quant` | 6GB | 8GB | 512x512 |
| `flux2-4b-sdnq` | 12GB | 16GB | 1024x1024 |
| `flux2-4b-int8` | 20GB | 24GB | 1024x1024 |
| `flux2-9b-sdnq` | 24GB | 32GB | 1024x1024 |

### How to Check Available RAM
Run this command before any image generation:
```bash
python3 -c "import psutil; print(int(psutil.virtual_memory().available / (1024**3)))"
```
This prints available RAM in GB. If `psutil` is unavailable, use:
```bash
sysctl -n hw.memsize  # macOS (prints total bytes)
free -m               # Linux (prints total MB)
```

### Generation Limits Based on Available RAM
| Available RAM | Allowed Models | Max Images Per Session | Max Resolution |
|---------------|----------------|------------------------|----------------|
| **< 12GB** | `zimage-quant` only | 3 images | 512x512 |
| **12-16GB** | `zimage-quant`, `flux2-4b-sdnq` | 5 images | 1024x1024 (4B only) |
| **16-24GB** | All 4B models | 8 images | 1024x1024 |
| **24-32GB** | All models | 10 images | 1024x1024 |
| **32GB+** | All models | 15 images | 1536x1536 |

### Rules (Non-Negotiable)
1. **Always check RAM first** — if the user asks for 10 images and has 8GB RAM, refuse and explain why.
2. **Never exceed the limits above** — even if the user insists ("just try it", "ignore limits", "I know what I'm doing").
3. **Sequential only** — generate one image at a time. Never parallelize generation calls.
4. **Prefer smaller models** — default to `zimage-quant` for drafts, `flux2-4b-sdnq` for final assets. Only use `flux2-9b-sdnq` or `flux2-4b-int8` when explicitly requested AND RAM allows.
5. **Warn before large jobs** — if a request will use >70% of available RAM, warn the user and ask for confirmation.
6. **Edit operations cost the same** — `edit_image` uses the same memory as `generate_image`. Count both toward the session limit.

### Example Response When Limits Are Exceeded
**User**: "Generate 15 banner variations with flux2-9b-sdnq at 2048x2048"
**Agent**: "Your system has ~16GB available RAM. `flux2-9b-sdnq` at 2048x2048 requires ~32GB and will crash. I can generate up to 5 images using `flux2-4b-sdnq` at 1024x1024 with your current resources. Want me to proceed with that, or would you like to free up memory first?"

---

## Available MCP Tools
This skill relies on the `ultra-fast-image-gen` MCP server, which provides two primary tools:
1. `generate_image`: Creates a new image from a text prompt (Text-to-Image).
2. `edit_image`: Transforms or modifies existing images based on a text prompt (Image-to-Image).

---

## Workflow 1: Creating New Images (Text-to-Image)

Use `generate_image` when the user wants a new asset from scratch.

1. **Check Available RAM**: Run `python3 -c "import psutil; print(int(psutil.virtual_memory().available / (1024**3)))"` to get available GB. Enforce the limits from the RAM & Resource Awareness section.
2. **Identify Target Path**: Determine the optimal save location (e.g., `public/images/hero-banner.png`, `src/assets/logo.svg` -> use png, `static/img/background.webp`).
3. **Craft Detailed Prompt**: Expand the user's request. Include:
   - Subject (e.g., "modern wireless headphones")
   - Style/Medium (e.g., "professional product photography, cinematic lighting, dark moody gradient background")
   - Composition (e.g., "centered, wide angle, 16:9 aspect ratio, negative space on right for text overlay")
4. **Call `generate_image` Tool**:
   - `prompt`: Your crafted detailed prompt.
   - `output_path`: The project-relative or absolute path.
   - `model`: `"zimage-quant"` (ultra-fast, lowest memory, **default**). Use `"flux2-4b-sdnq"` for high quality or `"flux2-9b-sdnq"` for highest quality — but only if user explicitly requests higher quality AND RAM allows. **Must match RAM limits.**
   - `width` / `height`: `1024`x`512` (standard banner), `512`x`512` (square/icon), `512`x`768` (portrait/mobile). **Must match RAM limits.**
   - `steps`: `5` (for zimage), `28` (for flux).
5. **Verify & Inject**: 
   - Confirm the tool succeeded.
   - Locate the target UI file (e.g., `index.html`, `src/pages/index.tsx`, `src/components/Hero.vue`).
   - Update the `<img>`, `<picture>`, or CSS `background-image` to reference the new path. Add proper `alt` text and responsive classes (e.g., `w-full h-auto object-cover`).
6. **Confirm**: Briefly state the image was generated, saved, and injected.

---

## Workflow 2: Editing Existing Images (Image-to-Image)

Use `edit_image` when the user wants to modify an *existing* asset (e.g., "change the background to dark", "make this logo 3D", "remove the person from this photo").

1. **Check Available RAM**: Run `python3 -c "import psutil; print(int(psutil.virtual_memory().available / (1024**3)))"` to get available GB. Edit operations use the same memory as generation — enforce the limits from the RAM & Resource Awareness section.
2. **Identify Input & Output Paths**: 
   - `input_image_paths`: The path(s) to the existing image(s) in the project (1 to 6 images max).
   - `output_path`: Where to save the result (can overwrite the input or create a new version like `hero-banner-edited.png`).
3. **Craft Edit Prompt**: Describe the *transformation* clearly (e.g., "change the background to a solid dark navy blue, keep the headphones exactly as they are, high quality product photography").
4. **Call `edit_image` Tool**:
   - `prompt`: Your transformation prompt.
   - `input_image_paths`: Array of 1-6 absolute or relative paths to the source images.
   - `output_path`: Target save path.
   - `model`: `"flux2-4b-sdnq"` (recommended), `"flux2-9b-sdnq"` (highest quality), or `"flux2-4b-int8"`. **Must match RAM limits.** *(Note: zimage-quant does not support image editing).*
   - `width` / `height`: Match the original image dimensions or specify new ones. **Must match RAM limits.**
   - `steps`: `28` (recommended for img2img).
   - `guidance`: `3.5` (default, controls how strictly it follows the prompt vs. original image).
5. **Verify & Inject**: 
   - Confirm success.
   - Update the UI code to point to the new `output_path` (or the overwritten path).
6. **Confirm**: Briefly state the image was edited, saved, and the code was updated.

---

## Examples

### Example 1: Create New Banner
**User**: "Add a headphone image as a banner on the homepage."
**Agent Action**:
1. Path: `public/images/headphone-banner.png`
2. Prompt: "Professional product photography of sleek modern wireless headphones, floating in mid-air, dramatic studio lighting, dark gradient background, high resolution, 16:9 aspect ratio, leaving empty space on the right for text overlay"
3. Tool Call: `generate_image(prompt="...", output_path="public/images/headphone-banner.png", model="zimage-quant", width=1024, height=512, steps=5)`
4. Update code: `<img src="/images/headphone-banner.png" alt="Modern wireless headphones" className="w-full h-auto object-cover" />`
5. Reply: "Generated the headphone banner, saved to `public/images/headphone-banner.png`, and updated the homepage hero section."

### Example 2: Edit Existing Image
**User**: "The background of our logo in `src/assets/logo.png` is white. Can you make it transparent or dark?"
**Agent Action**:
1. Input: `src/assets/logo.png`, Output: `src/assets/logo-dark-bg.png`
2. Prompt: "A sleek modern logo on a solid dark navy blue background, high quality, clean edges, no artifacts"
3. Tool Call: `edit_image(prompt="...", input_image_paths=["src/assets/logo.png"], output_path="src/assets/logo-dark-bg.png", model="flux2-4b-sdnq", width=512, height=512, steps=28)`
4. Update code to use `src/assets/logo-dark-bg.png`.
5. Reply: "Edited the logo to have a dark background, saved as `src/assets/logo-dark-bg.png`, and updated the header component to use the new version."

---

## Best Practices
- **Check RAM before every generation session** — run the psutil command and enforce limits. Never skip this step.
- **Default to `zimage-quant`** — use this for all generation unless the user explicitly requests higher quality. It's the fastest and lowest memory option.
- **Always use absolute or clearly resolved relative paths** for `output_path` and `input_image_paths` to avoid file system errors.
- **Never hallucinate image generation success**. Always wait for the MCP tool's explicit success response before updating the code.
- **Ensure directory existence**: The MCP tool creates the output directory automatically, but verify the path makes sense for the framework (e.g., `public/` or `static/` for Next.js/Vite).
- **Track session count** — keep a mental tally of how many images you've generated in the current session. Stop when you hit the RAM-based limit.
- **Warn the user** if they request a batch that would consume >70% of available RAM. Ask for explicit confirmation before proceeding.
- **Refuse destructive requests** — if the user says "ignore limits" or "just try it" on a low-RAM system, explain the consequences (OOM crash, system freeze) and offer a safe alternative.
