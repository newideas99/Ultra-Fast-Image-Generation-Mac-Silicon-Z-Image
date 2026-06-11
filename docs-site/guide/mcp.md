# MCP Server (AI Coding Agents)

The repository doubles as a local [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server. This allows AI coding assistants (OpenCode, Claude Desktop, Cursor, Claude Code) to generate and edit project assets — hero banners, icons, backgrounds — locally without relying on cloud APIs.

It exposes two tools:
- `generate_image`: Create new images from text prompts.
- `edit_image`: Transform 1-6 existing reference images based on a text prompt.

Both tools run the same `generate.py` CLI under the hood.

## Prerequisites

Install the `mcp` package into the app's environment first:

```bash
venv/bin/pip install mcp
# or, if using uv:
uv sync --extra mcp
```

## Configuration

### OpenCode (1-Click Setup)

Run the provided installation script:

```bash
python3 scripts/install-opencode-mcp.py
```

This script registers the server in `~/.config/opencode/opencode.json` and installs a "website-visual-assets" skill. It reuses `HF_TOKEN` from your environment or the repo `.env`, prompting only if neither exists.

### Manual Configuration

**OpenCode** (`~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "ultra-fast-image-gen": {
      "type": "local",
      "command": ["/path/to/ultra-fast-image-gen/venv/bin/python", "/path/to/ultra-fast-image-gen/mcp_server.py"],
      "environment": { "HF_TOKEN": "hf_..." },
      "timeout": 3600000
    }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ultra-fast-image-gen": {
      "command": "/path/to/ultra-fast-image-gen/venv/bin/python",
      "args": ["/path/to/ultra-fast-image-gen/mcp_server.py"],
      "env": { "HF_TOKEN": "hf_..." }
    }
  }
}
```

## Agent Instructions

To ensure your AI assistant automatically uses these tools, add the following to your project's `CLAUDE.md`, `.opencode/agents.md`, or `.cursorrules`:

```markdown
When asked to create or modify visual assets (e.g., "generate a hero banner", "make the logo background dark"), use the `generate_image` / `edit_image` tools from the `ultra-fast-image-gen` MCP server. 

- The default model `"zimage-quant"` is ultra-fast and lowest-memory.
- Use `"flux2-4b-sdnq"` or `"flux2-9b-sdnq"` only when higher quality is explicitly requested.
- Save outputs to logical project paths (e.g., `public/images/hero.png`).
```

> **Note:** No `HF_TOKEN` is needed for the standard models (Z-Image, FLUX.2-klein, Anima) — weights download to the local Hugging Face cache on first use. The token is only required for the gated uncensored text encoder.
> 
> **Heads-up:** An MCP generation and a web-UI generation each load their own model copy, so running both simultaneously will double memory usage.

---

**Next:** [Benchmarks](/guide/benchmarks)
