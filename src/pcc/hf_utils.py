import logging
import pathlib
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .layernorm_fusion import fuse_modules, replace_layers
from .model_adapter import ModelAdapter, SlicingConfig
from .rotate import slice_rotated_model


def do_not_initialize(func):

    def skip(*args, **kwargs) -> None:
        pass

    def wrapper(*args, **kwargs):
        kaiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kaiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn

        return result

    return wrapper


@do_not_initialize
def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    uninitialized: bool = False,
    dtype: torch.dtype = torch.float16,
    token: str | bool | None = None,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:

    model_type = "uninitialized" if uninitialized else "pretrained"
    local_model = model_path is not None

    if local_model and uninitialized:
        local_model = (pathlib.Path(model_path) / "config.json").exists()

    if not local_model:
        model_path = model_name

    logging.info(
        f"Loading %s config %s from %s",
        model_name,
        "and model weights" if not uninitialized else "",
        model_path if local_model else 'Hugging Face',
    )

    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=model_path,
        model_type=model_type,
        dtype=dtype,
        local_files_only=local_model,
        token=token,
    )

    model = model_adapter.model
    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token, local_files_only=local_model)

    model_adapter.post_init(tokenizer)
    logging.info("Loading model done")

    return model_adapter, tokenizer


@do_not_initialize
def load_sliced_model(
    model_name: str,
    sliced_model_path: str,
    *,
    token: str | None = None,
    lora_config: Any = None,
    sparsity: float | None = None,
    round_interval: int | None = 1,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
   
    my_model_suffix = pathlib.Path(model_name).name
    my_sliced_model_name = f"{my_model_suffix}_{sparsity}.pt"
    my_sliced_model_config = f"{my_model_suffix}_{sparsity}.json"

    model_adapter, tokenizer = get_model_and_tokenizer(
        model_name,
        model_path=sliced_model_path,
        uninitialized=True,
        token=token,
    )
    replace_layers(model_adapter)
    fuse_modules(model_adapter)

    hidden_size = model_adapter.hidden_size
    for layer_adapter in model_adapter.get_layers():
        if not model_adapter.parallel_blocks:
            layer_adapter.layer.mlp_shortcut_Q = torch.nn.Parameter(
                torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
            )
        layer_adapter.layer.attn_shortcut_Q = torch.nn.Parameter(
            torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
        )

    config_path = pathlib.Path(sliced_model_path) / my_sliced_model_config

    if config_path.exists():
        model_adapter.slicing_conf = SlicingConfig.from_json_string(config_path.read_text())

    if model_adapter.slicing_conf is None:
        new_embedding_dimension = int((1 - sparsity) * hidden_size)
        new_embedding_dimension -= new_embedding_dimension % round_interval
        config = SlicingConfig()
        config.const_dimension = new_embedding_dimension
        model_adapter.slicing_conf = config

    slice_rotated_model(model_adapter)

    if lora_config:
        from peft import get_peft_model

        model_adapter.model = get_peft_model(model_adapter.model, lora_config)

    logging.info(f"Loading sliced model weights from {sliced_model_path}")
    model_adapter.model.load_state_dict(
        torch.load(str(pathlib.Path(sliced_model_path) / my_sliced_model_name), map_location="cpu")
    )
    model_adapter.model.eval()

    return model_adapter, tokenizer
