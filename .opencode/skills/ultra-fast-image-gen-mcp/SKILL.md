# Ultra Fast Image Gen MCP (Image Generation & Editing)

Use this skill when the user asks to create, generate, edit, transform, or add visual assets (images, banners, icons, backgrounds) to a website project.

## Available MCP Tools
This skill relies on the `ultra-fast-image-gen` MCP server, which provides two primary tools:
1. `generate_image`: Creates a new image from a text prompt (Text-to-Image).
2. `edit_image`: Transforms or modifies existing images based on a text prompt (Image-to-Image).

---

## Workflow 1: Creating New Images (Text-to-Image)

Use `generate_image` when the user wants a new asset from scratch.

1. **Identify Target Path**: Determine the optimal save location (e.g., `public/images/hero-banner.png`, `src/assets/logo.svg` -> use png, `static/img/background.webp`).
2. **Craft Detailed Prompt**: Expand the user's request. Include:
   - Subject (e.g., "modern wireless headphones")
   - Style/Medium (e.g., "professional product photography, cinematic lighting, dark moody gradient background")
   - Composition (e.g., "centered, wide angle, 16:9 aspect ratio, negative space on right for text overlay")
3. **Call `generate_image` Tool**:
   - `prompt`: Your crafted detailed prompt.
   - `output_path`: The project-relative or absolute path.
   - `model`: `"zimage-quant"` (fastest, good for drafts/icons), `"flux2-4b-sdnq"` (high quality, recommended for final assets), or `"flux2-9b-sdnq"` (highest quality).
   - `width` / `height`: `1024`x`512` (standard banner), `512`x`512` (square/icon), `512`x`768` (portrait/mobile).
   - `steps`: `5` (for zimage), `28` (for flux).
4. **Verify & Inject**: 
   - Confirm the tool succeeded.
   - Locate the target UI file (e.g., `index.html`, `src/pages/index.tsx`, `src/components/Hero.vue`).
   - Update the `<img>`, `<picture>`, or CSS `background-image` to reference the new path. Add proper `alt` text and responsive classes (e.g., `w-full h-auto object-cover`).
5. **Confirm**: Briefly state the image was generated, saved, and injected.

---

## Workflow 2: Editing Existing Images (Image-to-Image)

Use `edit_image` when the user wants to modify an *existing* asset (e.g., "change the background to dark", "make this logo 3D", "remove the person from this photo").

1. **Identify Input & Output Paths**: 
   - `input_image_paths`: The path(s) to the existing image(s) in the project (1 to 6 images max).
   - `output_path`: Where to save the result (can overwrite the input or create a new version like `hero-banner-edited.png`).
2. **Craft Edit Prompt**: Describe the *transformation* clearly (e.g., "change the background to a solid dark navy blue, keep the headphones exactly as they are, high quality product photography").
3. **Call `edit_image` Tool**:
   - `prompt`: Your transformation prompt.
   - `input_image_paths`: Array of 1-6 absolute or relative paths to the source images.
   - `output_path`: Target save path.
   - `model`: `"flux2-4b-sdnq"` (recommended), `"flux2-9b-sdnq"` (highest quality), or `"flux2-4b-int8"`. *(Note: zimage-quant does not support image editing).*
   - `width` / `height`: Match the original image dimensions or specify new ones.
   - `steps`: `28` (recommended for img2img).
   - `guidance`: `3.5` (default, controls how strictly it follows the prompt vs. original image).
4. **Verify & Inject**: 
   - Confirm success.
   - Update the UI code to point to the new `output_path` (or the overwritten path).
5. **Confirm**: Briefly state the image was edited, saved, and the code was updated.

---

## Examples

### Example 1: Create New Banner
**User**: "Add a headphone image as a banner on the homepage."
**Agent Action**:
1. Path: `public/images/headphone-banner.png`
2. Prompt: "Professional product photography of sleek modern wireless headphones, floating in mid-air, dramatic studio lighting, dark gradient background, high resolution, 16:9 aspect ratio, leaving empty space on the right for text overlay"
3. Tool Call: `generate_image(prompt="...", output_path="public/images/headphone-banner.png", model="flux2-4b-sdnq", width=1024, height=512, steps=28)`
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
- **Always use absolute or clearly resolved relative paths** for `output_path` and `input_image_paths` to avoid file system errors.
- **Prefer `flux2-4b-sdnq`** for final website assets due to its superior prompt adherence and quality. Use `zimage-quant` only for rapid prototyping or simple icons.
- **Never hallucinate image generation success**. Always wait for the MCP tool's explicit success response before updating the code.
- **Ensure directory existence**: The MCP tool creates the output directory automatically, but verify the path makes sense for the framework (e.g., `public/` or `static/` for Next.js/Vite).
