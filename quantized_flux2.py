# Copyright 2024 The HuggingFace Team. All rights reserved.
# Adapted for FLUX.2-klein by adding Flux2Transformer2DModel and Qwen3 support

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Union

from huggingface_hub import ModelHubMixin, snapshot_download

from optimum.quanto import freeze, qtype, quantization_map, quantize, requantize, Optimizer
from optimum.quanto.models import is_diffusers_available

from diffusers.models.model_loading_utils import load_state_dict
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    _get_checkpoint_shard_files,
    is_accelerate_available,
)
from optimum.quanto.models.shared_dict import ShardedStateDict


class QuantizedDiffusersModel(ModelHubMixin):
    """Base class for quantized diffusers models."""
    BASE_NAME = "quanto"
    base_class = None

    def __init__(self, model: ModelMixin):
        if not isinstance(model, ModelMixin) or len(quantization_map(model)) == 0:
            raise ValueError("The source model must be a quantized diffusers model.")
        self._wrapped = model

    def __getattr__(self, name: str) -> Any:
        """If an attribute is not found in this class, look in the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = self.__dict__["_wrapped"]
            return getattr(wrapped, name)

    def forward(self, *args, **kwargs):
        return self._wrapped.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._wrapped.forward(*args, **kwargs)

    @staticmethod
    def _qmap_name():
        return f"{QuantizedDiffusersModel.BASE_NAME}_qmap.json"

    @classmethod
    def quantize(
        cls,
        model: ModelMixin,
        weights: Optional[Union[str, qtype]] = None,
        activations: Optional[Union[str, qtype]] = None,
        optimizer: Optional[Optimizer] = None,
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
    ):
        """Quantize the specified model."""
        if not isinstance(model, ModelMixin):
            raise ValueError("The source model must be a diffusers model.")

        quantize(
            model, weights=weights, activations=activations, optimizer=optimizer, include=include, exclude=exclude
        )
        freeze(model)
        return cls(model)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        if cls.base_class is None:
            raise ValueError("The `base_class` attribute needs to be configured.")

        if not is_accelerate_available():
            raise ValueError("Reloading a quantized diffusers model requires the accelerate library.")
        from accelerate import init_empty_weights

        if os.path.isdir(pretrained_model_name_or_path):
            working_dir = pretrained_model_name_or_path
        else:
            working_dir = snapshot_download(pretrained_model_name_or_path, **kwargs)

        # Look for a quantization map
        qmap_path = os.path.join(working_dir, cls._qmap_name())
        if not os.path.exists(qmap_path):
            raise ValueError(
                f"No quantization map found in {pretrained_model_name_or_path}: is this a quantized model ?"
            )

        # Look for original model config file.
        model_config_path = os.path.join(working_dir, CONFIG_NAME)
        if not os.path.exists(model_config_path):
            raise ValueError(f"{CONFIG_NAME} not found in {pretrained_model_name_or_path}.")

        with open(qmap_path, "r", encoding="utf-8") as f:
            qmap = json.load(f)

        with open(model_config_path, "r", encoding="utf-8") as f:
            original_model_cls_name = json.load(f)["_class_name"]
        configured_cls_name = cls.base_class.__name__
        if configured_cls_name != original_model_cls_name:
            raise ValueError(
                f"Configured base class ({configured_cls_name}) differs from what was derived from the provided configuration ({original_model_cls_name})."
            )

        # Create an empty model
        config = cls.base_class.load_config(pretrained_model_name_or_path, **kwargs)
        with init_empty_weights():
            model = cls.base_class.from_config(config)

        # Look for the index of a sharded checkpoint
        checkpoint_file = os.path.join(working_dir, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.exists(checkpoint_file):
            # Convert the checkpoint path to a list of shards
            _, sharded_metadata = _get_checkpoint_shard_files(working_dir, checkpoint_file)
            # Create a mapping for the sharded safetensor files
            state_dict = ShardedStateDict(working_dir, sharded_metadata["weight_map"])
        else:
            # Look for a single checkpoint file
            checkpoint_file = os.path.join(working_dir, SAFETENSORS_WEIGHTS_NAME)
            if not os.path.exists(checkpoint_file):
                raise ValueError(f"No safetensor weights found in {pretrained_model_name_or_path}.")
            # Get state_dict from model checkpoint
            state_dict = load_state_dict(checkpoint_file)

        # Requantize and load quantized weights from state_dict
        requantize(model, state_dict=state_dict, quantization_map=qmap)
        model.eval()
        return cls(model)

    def _save_pretrained(self, save_directory: Path) -> None:
        self._wrapped.save_pretrained(save_directory)
        # Save quantization map to be able to reload the model
        qmap_name = os.path.join(save_directory, self._qmap_name())
        qmap = quantization_map(self._wrapped)
        with open(qmap_name, "w", encoding="utf8") as f:
            json.dump(qmap, f, indent=4)


# Import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel


class QuantizedFlux2Transformer2DModel(QuantizedDiffusersModel):
    """Quantized FLUX.2 Transformer model."""
    base_class = Flux2Transformer2DModel
