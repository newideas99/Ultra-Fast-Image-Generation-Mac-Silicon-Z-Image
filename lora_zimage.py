"""
LoRA (Low-Rank Adaptation) support for Z-Image on Mac.

This module provides a simplified LoRA implementation that works with
quantized models on Apple Silicon (MPS). LoRA weights are applied as
separate low-rank matrices that add to the model's layer outputs.
"""

import math
import weakref
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file


# Module types that can have LoRA applied
LINEAR_MODULES = ['Linear', 'LoRACompatibleLinear', 'QLinear']
CONV_MODULES = ['Conv2d', 'LoRACompatibleConv', 'QConv2d']


class LoRAModule(nn.Module):
    """
    A single LoRA module that wraps an existing layer.

    LoRA works by adding a low-rank decomposition (A @ B) to the original
    layer output: output = original_output + (x @ A @ B) * scale
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        lora_dim: int = 4,
        alpha: float = 1.0,
        multiplier: float = 1.0,
        network: 'LoRANetwork' = None,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.org_module_ref = weakref.ref(org_module)
        self.network_ref = weakref.ref(network) if network else None
        self.multiplier = multiplier
        self.lora_dim = lora_dim

        # Determine input/output dimensions based on module type
        if org_module.__class__.__name__ in CONV_MODULES:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(in_dim, lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        # Calculate scale factor (alpha / rank)
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Initialize weights (kaiming for down, zeros for up)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Store original forward and replace it
        self.org_forward = org_module.forward
        self.org_module = [org_module]  # Use list to avoid registering as submodule

    def apply_to(self):
        """Hook this LoRA into the original module's forward pass."""
        self.org_module[0].forward = self.forward

    def remove(self):
        """Restore the original forward pass."""
        self.org_module[0].forward = self.org_forward

    def forward(self, x, *args, **kwargs):
        """Forward pass: original output + LoRA contribution."""
        # Get network to check if active
        network = self.network_ref() if self.network_ref else None

        # If network is not active, just use original forward
        if network and not network.is_active:
            return self.org_forward(x, *args, **kwargs)

        # Get original output
        org_output = self.org_forward(x, *args, **kwargs)

        # Get multiplier from network or use local
        multiplier = network.multiplier if network else self.multiplier
        if multiplier == 0:
            return org_output

        # Compute LoRA contribution
        # Need to handle potential dtype mismatches with quantized models
        lora_input = x
        if hasattr(x, 'dequantize'):
            lora_input = x.dequantize()

        # Cast to LoRA weight dtype for computation
        lora_dtype = self.lora_down.weight.dtype
        if lora_input.dtype != lora_dtype:
            lora_input = lora_input.to(lora_dtype)

        lora_output = self.lora_up(self.lora_down(lora_input))
        lora_output = lora_output * self.scale * multiplier

        # Cast back to original output dtype
        if lora_output.dtype != org_output.dtype:
            lora_output = lora_output.to(org_output.dtype)

        return org_output + lora_output


class LoRANetwork(nn.Module):
    """
    A network of LoRA modules that can be applied to a transformer model.

    Supports loading LoRA weights from safetensors files in various formats
    (standard LoRA, PEFT format, etc.)
    """

    # Target modules for Z-Image transformer
    TARGET_MODULES = ["ZImageTransformer2DModel"]

    def __init__(
        self,
        transformer: nn.Module,
        lora_dim: int = 4,
        alpha: float = 1.0,
        multiplier: float = 1.0,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.lora_dim = lora_dim
        self.alpha = alpha
        self._multiplier = multiplier
        self.is_active = False
        self.lora_modules: List[LoRAModule] = []

        target_modules = target_modules or self.TARGET_MODULES

        # Create LoRA modules for each target layer
        self._create_modules(transformer, target_modules)

        print(f"Created LoRA network with {len(self.lora_modules)} modules (dim={lora_dim}, alpha={alpha})")

    def _create_modules(self, root_module: nn.Module, target_replace_modules: List[str]):
        """Create LoRA modules for all matching layers in the model."""
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ in LINEAR_MODULES
                    is_conv = child_module.__class__.__name__ in CONV_MODULES

                    if is_linear or is_conv:
                        # Create unique name for this LoRA
                        lora_name = f"transformer.{name}.{child_name}".replace("..", ".")
                        lora_name = lora_name.replace(".", "_")

                        lora = LoRAModule(
                            lora_name=lora_name,
                            org_module=child_module,
                            lora_dim=self.lora_dim,
                            alpha=self.alpha,
                            multiplier=self._multiplier,
                            network=self,
                        )
                        self.lora_modules.append(lora)
                        # Register as submodule for proper device/dtype handling
                        self.add_module(lora_name, lora)

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: float):
        self._multiplier = value

    def apply_to(self):
        """Apply all LoRA modules to their target layers."""
        for lora in self.lora_modules:
            lora.apply_to()
        self.is_active = True
        print(f"LoRA applied to {len(self.lora_modules)} modules")

    def remove(self):
        """Remove all LoRA modules from their target layers."""
        for lora in self.lora_modules:
            lora.remove()
        self.is_active = False
        print("LoRA removed from all modules")

    def __enter__(self):
        """Context manager entry - activate LoRA."""
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deactivate LoRA."""
        self.is_active = False

    def load_weights(self, weights: Union[str, Dict[str, torch.Tensor]]):
        """
        Load LoRA weights from a safetensors file or state dict.

        Supports multiple formats:
        - Standard LoRA format (lora_down/lora_up)
        - PEFT format (lora_A/lora_B)
        - Various naming conventions
        """
        if isinstance(weights, str):
            print(f"Loading LoRA weights from {weights}")
            weights = load_file(weights)

        # Detect format and convert keys
        converted_weights = self._convert_weight_keys(weights)

        # Build mapping from weight keys to our module names
        current_state = self.state_dict()
        load_dict = {}
        missing_keys = []

        for key, value in converted_weights.items():
            if key in current_state:
                # Handle shape mismatches (expanding/shrinking LoRA)
                if value.shape != current_state[key].shape:
                    value = self._resize_weight(value, current_state[key].shape, key)
                load_dict[key] = value
            else:
                missing_keys.append(key)

        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys not found in network")

        # Load the weights
        info = self.load_state_dict(load_dict, strict=False)
        print(f"Loaded {len(load_dict)} weight tensors")

        return info

    def _convert_weight_keys(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert weight keys from various formats to our internal format."""
        converted = {}

        for key, value in weights.items():
            new_key = key

            # PEFT format: lora_A -> lora_down, lora_B -> lora_up
            new_key = new_key.replace('lora_A', 'lora_down')
            new_key = new_key.replace('lora_B', 'lora_up')

            # Handle various prefix formats
            # transformer.xxx -> transformer_xxx
            new_key = new_key.replace('.', '_')

            # Handle $$ separators (from zimagetrain format)
            new_key = new_key.replace('$$', '_')

            # Fix up the lora_down/lora_up suffixes
            new_key = new_key.replace('_lora_down_', '.lora_down.')
            new_key = new_key.replace('_lora_up_', '.lora_up.')

            # Skip alpha tensors (we compute scale from dim)
            if key.endswith('.alpha') or key.endswith('_alpha'):
                continue

            converted[new_key] = value

        return converted

    def _resize_weight(
        self,
        src_weight: torch.Tensor,
        tgt_shape: tuple,
        key: str
    ) -> torch.Tensor:
        """Resize LoRA weight to match target shape (for model compatibility)."""
        if len(src_weight.shape) != 2:
            # Only handle linear layers for now
            return src_weight

        src_h, src_w = src_weight.shape
        tgt_h, tgt_w = tgt_shape

        if (src_h, src_w) == (tgt_h, tgt_w):
            return src_weight

        new_weight = torch.zeros(tgt_shape, dtype=src_weight.dtype, device=src_weight.device)

        # Copy what we can
        copy_h = min(src_h, tgt_h)
        copy_w = min(src_w, tgt_w)
        new_weight[:copy_h, :copy_w] = src_weight[:copy_h, :copy_w]

        print(f"Resized {key}: {src_weight.shape} -> {tgt_shape}")
        return new_weight


def load_lora_for_pipeline(
    pipe,
    lora_path: str,
    lora_dim: Optional[int] = None,
    alpha: Optional[float] = None,
    multiplier: float = 1.0,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> LoRANetwork:
    """
    Load a LoRA adapter for a Z-Image pipeline.

    Args:
        pipe: The ZImagePipeline instance
        lora_path: Path to the LoRA safetensors file
        lora_dim: LoRA dimension (auto-detected if None)
        alpha: LoRA alpha (defaults to lora_dim if None)
        multiplier: LoRA strength multiplier (default 1.0)
        device: Device to load LoRA to (defaults to pipe's device)
        dtype: Dtype for LoRA weights (defaults to float32 for MPS compatibility)

    Returns:
        LoRANetwork instance (already applied to the model)
    """
    # Load weights to detect dimension
    weights = load_file(lora_path)

    # Auto-detect lora_dim from weights
    if lora_dim is None:
        for key, value in weights.items():
            if 'lora_down' in key.lower() or 'lora_a' in key.lower():
                if len(value.shape) == 2:
                    lora_dim = value.shape[0]
                    break
                elif len(value.shape) == 4:  # Conv
                    lora_dim = value.shape[0]
                    break

    if lora_dim is None:
        raise ValueError("Could not auto-detect LoRA dimension from weights")

    if alpha is None:
        alpha = lora_dim

    print(f"Detected LoRA dim={lora_dim}, alpha={alpha}")

    # Create the network
    network = LoRANetwork(
        transformer=pipe.transformer,
        lora_dim=lora_dim,
        alpha=alpha,
        multiplier=multiplier,
    )

    # Move to device/dtype
    device = device or pipe.device
    dtype = dtype or torch.float32  # float32 is safer for MPS
    network.to(device=device, dtype=dtype)

    # Load the weights
    network.load_weights(weights)

    # Apply to the model
    network.apply_to()
    network.is_active = True

    return network


def list_lora_files(directory: str) -> List[str]:
    """List all safetensors files in a directory that could be LoRA weights."""
    import os

    lora_files = []
    if os.path.isdir(directory):
        for f in os.listdir(directory):
            if f.endswith('.safetensors'):
                lora_files.append(os.path.join(directory, f))
    return sorted(lora_files)
