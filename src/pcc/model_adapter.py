from __future__ import annotations

import copy
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, final

import torch
from torch import FloatTensor, Tensor
from torch.nn import Linear, Module
from transformers import PreTrainedTokenizerBase


class LayerAdapter(ABC):

    @property
    @abstractmethod
    def layer(self) -> Module:

        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_args_position(self) -> int:
       
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_output_position(self) -> int:
        
        raise NotImplementedError

    @abstractmethod
    def get_first_layernorm(self) -> Module:
        
        raise NotImplementedError

    @abstractmethod
    def get_second_layernorm(self) -> Module:

        raise NotImplementedError

    @abstractmethod
    def get_attention_inputs(self) -> Sequence[Linear]:

        raise NotImplementedError

    @abstractmethod
    def get_attention_output(self) -> Linear:

        raise NotImplementedError

    @abstractmethod
    def get_mlp_inputs(self) -> Sequence[Linear]:

        raise NotImplementedError

    @abstractmethod
    def get_mlp_output(self) -> Linear:

        raise NotImplementedError

    def get_updated_args(self, hidden_states: Any, args: tuple) -> tuple:

        return (
            args[: self.hidden_states_args_position] + (hidden_states,) + args[self.hidden_states_args_position + 1 :]
        )


class ModelAdapter(ABC):


    def __init__(self):
        self.slicing_conf: SlicingConfig | None = None

    @property
    @abstractmethod
    def model(self) -> Module:

        raise NotImplementedError

    @property
    @abstractmethod
    def config(self) -> object:

        raise NotImplementedError

    @property
    @abstractmethod
    def config_type(self) -> type:

        raise NotImplementedError

    @property
    @abstractmethod
    def parallel_blocks(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def seqlen(self) -> int:

        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self) -> int:

        raise NotImplementedError

    @property
    @abstractmethod
    def should_bake_mean_into_linear(self) -> bool:

        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_type(self) -> type:

        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_norm_type(self) -> type:

        raise NotImplementedError

    @property
    @abstractmethod
    def layer_adapter_type(self) -> type:

        raise NotImplementedError

    @property
    @abstractmethod
    def compressed_layer_type(self) -> type:

        raise NotImplementedError

    @property
    @abstractmethod
    def use_cache(self) -> bool:
        raise NotImplementedError

    @use_cache.setter
    @abstractmethod
    def use_cache(self, value: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:

        raise NotImplementedError

    @abstractmethod
    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:

        raise NotImplementedError

    @abstractmethod
    def get_layers(self) -> Sequence[LayerAdapter]:

        raise NotImplementedError

    @abstractmethod
    def get_raw_layer_at(self, index: int) -> Module:

        raise NotImplementedError

    @abstractmethod
    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:

        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self) -> list[Module]:

        raise NotImplementedError

    @abstractmethod
    def get_pre_head_layernorm(self) -> Module:

        raise NotImplementedError

    @abstractmethod
    def get_lm_head(self) -> Linear:

        raise NotImplementedError

    @property
    def no_split_module_classes(self) -> list[str] | None:

        return [self.original_layer_type.__name__, self.compressed_layer_type.__name__]

    @final
    def convert_layer_to_compressed_and_register_buffers(self, layer: Module, layer_idx: int | None) -> Module:

        compressed_layer = self.convert_layer_to_compressed(layer, layer_idx)
        if not self.parallel_blocks:
            compressed_layer.register_parameter('mlp_shortcut_Q', None)
        compressed_layer.register_parameter('attn_shortcut_Q', None)
        return compressed_layer

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:

        pass

    @classmethod
    def from_model(
        cls,
        model_name: str,
        model_path: str,
        *,
        model_type: str = 'pretrained',
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter:
       

        def find_recursively(adapter_cls: type[ModelAdapter]) -> ModelAdapter | None:

            # depth first search to find the most specific subclass that can handle the model
            for subclass in adapter_cls.__subclasses__():
                candidate = find_recursively(subclass)
                if candidate is not None:
                    return candidate

            if inspect.isabstract(adapter_cls):
                return None

            return adapter_cls._from_model(
                model_name,
                model_path=model_path,
                model_type=model_type,
                dtype=dtype,
                local_files_only=local_files_only,
                token=token,
            )

        adapter = find_recursively(cls)
        if adapter is not None:
            return adapter

        raise NotImplementedError(f"{model_path} is neither a Hugging Face model nor a supported local model.")

    @classmethod
    def _from_model(
        cls,
        model_name: str,
        model_path: str,
        *,
        model_type: str = 'pretrained',
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        match model_type:
            case 'pretrained':
                return cls._from_pretrained(
                    model_name,
                    model_path=model_path,
                    dtype=dtype,
                    local_files_only=local_files_only,
                    token=token,
                )

            case 'uninitialized':
                return cls._from_uninitialized(
                    model_name,
                    model_path=model_path,
                    dtype=dtype,
                    local_files_only=local_files_only,
                    token=token,
                )
            case _:
                raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    @abstractmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:

        raise NotImplementedError


@dataclass
class SlicingConfig:

    hidden_size: int = 0
    layers_num: int = 0
    do_slice_head: bool = False
    parallel_blocks: bool = False

    embedding_dimensions: dict[int, int] = field(default_factory=dict)

    attention_input_dimensions: dict[int, int] = field(default_factory=dict)
    attention_output_dimensions: dict[int, int] = field(default_factory=dict)

    mlp_input_dimensions: dict[int, int] = field(default_factory=dict)
    mlp_output_dimensions: dict[int, int] = field(default_factory=dict)

    head_dimension: int | None = None

    const_dimension: int | None = None  # to be able to load models without config, sliced with const sparsity

    @staticmethod
    def from_dict(d: dict) -> 'SlicingConfig':

        def convert_dict_keys_to_int(d: Any) -> Any:
            # recursively convert all numeric string keys to int
            if not isinstance(d, dict):
                return d

            if all(isinstance(k, str) and k.isnumeric() for k in d.keys()):
                d = {int(k): v for k, v in d.items()}
            else:
                d = {k: convert_dict_keys_to_int(v) for k, v in d.items()}

            return d

        return SlicingConfig(**convert_dict_keys_to_int(d))

    @staticmethod
    def from_json_string(json_str: str) -> 'SlicingConfig':
        return SlicingConfig.from_dict(json.loads(json_str))

    def to_dict(self) -> dict:
        # workaround until 'dataclasses.asdict support defaultdict fields #32056' is in the Python release used
        self.embedding_dimensions = {k: v for k, v in self.embedding_dimensions.items()}

        return asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict())

    def clone(self) -> 'SlicingConfig':
        return copy.deepcopy(self)
