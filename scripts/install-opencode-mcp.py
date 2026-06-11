#!/usr/bin/env python3
"""
Surgically installs the Ultra Fast Image Gen MCP server and OpenCode skill globally.
Preserves existing configuration, handles JSONC comments, and maintains formatting.
"""

import os
import json
import re
import sys
import shutil
import getpass
from pathlib import Path

def strip_jsonc_comments(text: str) -> str:
    """Safely strip // and /* */ comments from JSONC for parsing."""
    pattern = r'("(?:\\.|[^"\\])*")|(/\*.*?\*/|//[^\r\n]*)'
    return re.sub(pattern, lambda m: m.group(1) or "", text, flags=re.DOTALL)

def main():
    config_dir = Path.home() / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "opencode.json"
    if not config_file.exists() and (config_dir / "opencode.jsonc").exists():
        config_file = config_dir / "opencode.jsonc"
        
    repo_root = (Path(__file__).parent.parent).resolve()
    mcp_script_path = (repo_root / "mcp_server.py").resolve()
    
    if not mcp_script_path.exists():
        print(f"❌ Error: mcp_server.py not found at {mcp_script_path}")
        sys.exit(1)

    mcp_name = "ultra-fast-image-gen"
    
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("\n🔑 Hugging Face API Token (HF_TOKEN) is required for model downloads.")
        hf_token = getpass.getpass("Enter your HF_TOKEN (input will be hidden): ").strip()
        if not hf_token:
            print("⚠️  Warning: No token provided. You will need to add it manually to the config later.")
            hf_token = "your_huggingface_token_here"

    # Use venv python if available to match Launch.command environment and cache
    venv_python = (repo_root / "venv" / "bin" / "python").resolve()
    python_cmd = str(venv_python) if venv_python.exists() else "python3"

    new_mcp_config = {
        "type": "local",
        "command": [python_cmd, str(mcp_script_path)],
        "environment": {
            "HF_TOKEN": hf_token
        },
        "timeout": 3600000  # 1 hour timeout for large model loads
    }

    # Install OpenCode Skill
    source_skill_dir = (Path(__file__).parent.parent / ".opencode" / "skills" / "ultra-fast-image-gen-mcp").resolve()
    target_skill_dir = Path.home() / ".config" / "opencode" / "skills" / "ultra-fast-image-gen-mcp"

    if source_skill_dir.exists():
        if target_skill_dir.exists():
            print(f"🔄 Updating existing skill: {target_skill_dir}")
            shutil.rmtree(target_skill_dir)
        shutil.copytree(source_skill_dir, target_skill_dir)
        print(f"✅ Installed OpenCode skill to: {target_skill_dir}")
    else:
        print(f"⚠️  Warning: Skill directory not found at {source_skill_dir}")

    if config_file.exists():
        print(f"📖 Reading existing config: {config_file}")
        try:
            raw_text = config_file.read_text(encoding="utf-8")
            clean_text = strip_jsonc_comments(raw_text)
            config = json.loads(clean_text)
            
            if config_file.suffix == ".jsonc":
                print("⚠️  Note: Updating .jsonc will normalize it to standard JSON (comments removed) to guarantee validity.")
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON/JSONC in {config_file}\n{e}")
            sys.exit(1)
    else:
        print(f"📄 Creating new config: {config_file}")
        config = {"$schema": "https://opencode.ai/config.json"}

    if "mcp" not in config:
        config["mcp"] = {}

    if mcp_name in config["mcp"]:
        print(f"🔄 MCP server '{mcp_name}' already exists. Updating configuration...")
    else:
        print(f"➕ Adding new MCP server: '{mcp_name}'")

    config["mcp"][mcp_name] = new_mcp_config

    # Set global experimental MCP timeout as a workaround for OpenCode timeout bugs
    if "experimental" not in config:
        config["experimental"] = {}
    if "mcp_timeout" not in config["experimental"]:
        config["experimental"]["mcp_timeout"] = 3600000
        print("⏱️  Set global experimental.mcp_timeout to 1 hour (3600000ms)")

    try:
        config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"✅ Successfully installed MCP server to {config_file}")
        print("\n💡 Tip: Restart OpenCode or run 'opencode mcp list' to verify.")
    except Exception as e:
        print(f"❌ Failed to write config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
