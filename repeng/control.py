import dataclasses
import typing
import warnings

import torch
from transformers import PretrainedConfig, PreTrainedModel

if typing.TYPE_CHECKING:
    from .extract import ControlVector


class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(self, model: PreTrainedModel, layer_ids: typing.Iterable[int]):
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new ControlModel around a model instance, initializing control on
        the layers specified in `layer_ids`.
        """

        super().__init__()
        self.model = model

        layers = model_layer_list(model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, ControlModule):
                layers[layer_id] = ControlModule(layer, model_type=model.config.model_type)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
        After using this method, `set_control` and `reset` will not work.
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layers[layer_id] = layers[layer_id].block
        return self.model

    def set_control(
        self, control: "ControlVector", coeff: float = 1.0, **kwargs
    ) -> None:
        """
        Set a `ControlVector` for the layers this ControlModel handles, with a strength given
        by `coeff`. (Negative `coeff` values invert the control vector, e.g. happiness→sadness.)
        `coeff` defaults to `1.0`.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        raw_control = {}
        for layer_id in self.layer_ids:
            raw_control[layer_id] = torch.tensor(
                coeff * control.directions[layer_id]
            ).to(self.model.device, dtype=self.model.dtype)
        self.set_raw_control(raw_control, **kwargs)

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: ControlModule = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@dataclasses.dataclass
class BlockControlParams:
    control: torch.Tensor | None = None
    normalize: bool = False
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()


class ControlModule(torch.nn.Module):
    def __init__(self, block: torch.nn.Module, model_type: str = None) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        self.params: BlockControlParams = BlockControlParams.default()
        self.model_type = model_type
        
        # Create a set of known attributes that might be accessed
        # This prevents AttributeError for common architecture-specific attributes
        self._attribute_defaults = {
            'attention_type': 'standard',  # Common in older models
            'layer_idx': None,  # Common in newer models
            'is_causal': True,  # Common attention attribute
            'rope_theta': 10000.0,  # RoPE parameter
            'max_position_embeddings': 2048,  # Position encoding
        }

    def set_control(self, params: BlockControlParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(BlockControlParams.default())

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output
        elif len(control.shape) == 1:
            control = control.reshape(1, 1, -1)

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)

        norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # we should ignore the padding tokens when doing the activation addition
        # mask has ones for non padding tokens and zeros at padding tokens.
        # only tested this on left padding
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            target_shape = modified.shape
            mask = (
                (col_indices >= zero_indices)
                .float()
                .reshape(target_shape[0], target_shape[1], 1)
            )
            mask = mask.to(modified.dtype).to(modified.device)
        else:
            mask = 1.0

        modified = self.params.operator(modified, control * mask)

        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output

    def __getattr__(self, name: str) -> typing.Any:
        # First check if it's one of our own attributes
        if name in ("block", "params", "model_type", "_attribute_defaults"):
            return super().__getattr__(name)
        
        # Try to get from the wrapped block
        try:
            return getattr(self.block, name)
        except AttributeError:
            # If the attribute doesn't exist on the block, check if we have a default
            if hasattr(self, '_attribute_defaults') and name in self._attribute_defaults:
                return self._attribute_defaults[name]
            
            # Log a debug warning for unknown attributes (can be removed in production)
            if not name.startswith('_'):  # Ignore private attributes
                warnings.warn(
                    f"Attribute '{name}' not found on {type(self.block).__name__}. "
                    f"This might indicate model incompatibility. Returning None.",
                    stacklevel=2
                )
            return None


def model_layer_list(model: ControlModel | PreTrainedModel) -> torch.nn.ModuleList:
    """
    Get the list of transformer layers from various model architectures.
    Supports multiple naming conventions used by different model families.
    """
    if isinstance(model, ControlModel):
        model = model.model

    # Extended list of possible layer paths for different architectures
    target_suffixes = [
        "repeng_layers",  # override
        "model.layers",  # llama, mistral, gemma, qwen2, qwen3, yi, deepseek, ...
        "transformer.h",  # gpt-2, gpt-j
        "transformer.blocks",  # mpt
        "gpt_neox.layers",  # gpt-neox
        "transformer.layers",  # gpt-neox alternative
        "model.decoder.layers",  # opt
        "bert.encoder.layer",  # bert
        "encoder.layer",  # bert alternative
        "h",  # some gpt2 variants
        "transformer.layer",  # some transformer models
        "blocks",  # some custom architectures
    ]
    
    for suffix in target_suffixes:
        candidates = [
            v
            for k, v in model.named_modules()
            if k.endswith(suffix) and isinstance(v, torch.nn.ModuleList)
        ]
        if len(candidates) == 1:
            return candidates[0]
    
    # If we still haven't found it, try a more flexible approach
    # Look for any ModuleList that contains transformer-like layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            # Check if the first element looks like a transformer layer
            # (has attention and/or mlp components)
            first_layer = module[0]
            has_attention = any(
                'attn' in n or 'attention' in n 
                for n, _ in first_layer.named_modules()
            )
            has_mlp = any(
                'mlp' in n or 'ffn' in n or 'feedforward' in n 
                for n, _ in first_layer.named_modules()
            )
            
            if has_attention or has_mlp:
                warnings.warn(
                    f"Using heuristic to find layers at '{name}'. "
                    f"Consider setting model.repeng_layers explicitly for better control."
                )
                return module

    raise ValueError(
        f"don't know how to get layer list for {type(model).__name__}! "
        f"Please set model.repeng_layers = model.<path_to_layers> before passing to ControlModel. "
        f"Common paths: model.layers, transformer.h, transformer.blocks"
    )
