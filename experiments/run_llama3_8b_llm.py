import argparse
import logging
import os
import pathlib
import shutil
import torch
import wandb
import pcc
from pcc import data_utils, gpu_utils, hf_utils
from pcc.config import config
def slicing_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="meta-llama/Llama-2-13b-hf",)
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument("--model-path",type=str,default=None,)
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")   
    parser.add_argument(
        "--cal-dataset",
        type=str,
        default="wikitext2",
    )  
    parser.add_argument("--cal-nsamples",type=int,default=128,)
    parser.add_argument("--cal-batch-size", type=int, default=16)
    parser.add_argument("--cal-max-seqlen", type=int, default=2048)
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--round-interval",type=int,default=8)
    parser.add_argument("--ppl-eval-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-eval-batch-size", type=int, default=1)
    parser.add_argument("--ppl-eval-nsamples", type=int, default=128)
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument('--wandb-project', type=str, default="pcc", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu',)
    return parser.parse_args() if interactive else parser.parse_args('')


def process_slicing_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')
    if args.device:
        config.device = torch.device(args.device)
    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")


def slicing_main(args: argparse.Namespace) -> None:
    logging.info("Running PCC cross-layer compensation on LLaMA3-8B.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
        args.model, args.model_path, token=args.hf_token, dtype=config.dtype
    )
    model = model_adapter.model
    dataset = data_utils.get_dataset(args.cal_dataset)
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )
    model.to('cuda:0')
    def merge_layers_return_model(model):
        #################layer23
        model.model.layers[23].self_attn.q_proj.weight.data=model.model.layers[30].self_attn.q_proj.weight.data
        model.model.layers[23].self_attn.k_proj.weight.data=model.model.layers[30].self_attn.k_proj.weight.data
        model.model.layers[23].self_attn.v_proj.weight.data=model.model.layers[30].self_attn.v_proj.weight.data
        model.model.layers[23].self_attn.o_proj.weight.data=model.model.layers[30].self_attn.o_proj.weight.data
        model.model.layers[23].mlp.gate_proj.weight.data=model.model.layers[30].mlp.gate_proj.weight.data
        model.model.layers[23].mlp.up_proj.weight.data=model.model.layers[30].mlp.up_proj.weight.data
        model.model.layers[23].mlp.down_proj.weight.data=model.model.layers[30].mlp.down_proj.weight.data
        model.model.layers[23].input_layernorm.weight.data=model.model.layers[30].input_layernorm.weight.data
        model.model.layers[23].post_attention_layernorm.weight.data=model.model.layers[30].post_attention_layernorm.weight.data
        #################layer24
        model.model.layers[24].self_attn.q_proj.weight.data=model.model.layers[31].self_attn.q_proj.weight.data
        model.model.layers[24].self_attn.k_proj.weight.data=model.model.layers[31].self_attn.k_proj.weight.data
        model.model.layers[24].self_attn.v_proj.weight.data=model.model.layers[31].self_attn.v_proj.weight.data
        model.model.layers[24].self_attn.o_proj.weight.data=model.model.layers[31].self_attn.o_proj.weight.data
        model.model.layers[24].mlp.gate_proj.weight.data=model.model.layers[31].mlp.gate_proj.weight.data
        model.model.layers[24].mlp.up_proj.weight.data=model.model.layers[31].mlp.up_proj.weight.data
        model.model.layers[24].mlp.down_proj.weight.data=model.model.layers[31].mlp.down_proj.weight.data
        model.model.layers[24].input_layernorm.weight.data=model.model.layers[31].input_layernorm.weight.data
        model.model.layers[24].post_attention_layernorm.weight.data=model.model.layers[31].post_attention_layernorm.weight.data
        
        for diff_lay in range(31,24,-1):
            del (model.model.layers[diff_lay])
        return model
    model=merge_layers_return_model(model)
        
    
    def create_hook_1():
        def zero_column_hook_1(module, input, output):
            hidden_states, *other_outputs = output
            hidden_states[:, :, 2352] = hidden_states[:, :, 2352]*1.4926-5.8043
            hidden_states[:, :, 2252] = hidden_states[:, :, 2252]*4.4202+1.5727
            hidden_states[:, :, 4055] = hidden_states[:, :, 4055]*0.1263-0.3875
            hidden_states[:, :, 2646] = hidden_states[:, :, 2646]*1.5769-0.5102
            hidden_states[:, :, 3026] = hidden_states[:, :, 3026]*1.8497+0.9916
            hidden_states[:, :, 1951] = hidden_states[:, :, 1951]*1.9491-0.3611
            hidden_states[:, :, 229] = hidden_states[:, :, 229]*1.7933-0.8872
            hidden_states[:, :, 2298] = hidden_states[:, :, 2298]*2.149+0.6941
            hidden_states[:, :, 3118] = hidden_states[:, :, 3118]*2.5083+0.7848
            hidden_states[:, :, 2995] = hidden_states[:, :, 2995]*1.0989+0.7165
            hidden_states[:, :, 1188] = hidden_states[:, :, 1188]*1.5236-0.7614
            hidden_states[:, :, 3139] = hidden_states[:, :, 3139]*1.6943+0.7245
            hidden_states[:, :, 2742] = hidden_states[:, :, 2742]*1.6747-0.1093
            hidden_states[:, :, 761] = hidden_states[:, :, 761]*1.7464+0.1719
            hidden_states[:, :, 3768] = hidden_states[:, :, 3768]*2.0833+0.6215
            hidden_states[:, :, 2209] = hidden_states[:, :, 2209]*1.8609+0.1599
            hidden_states[:, :, 3328] = hidden_states[:, :, 3328]*1.4896+0.6866
            hidden_states[:, :, 1658] = hidden_states[:, :, 1658]*1.7135+0.1595
            hidden_states[:, :, 504] = hidden_states[:, :, 504]*1.9167-0.0334
            hidden_states[:, :, 2992] = hidden_states[:, :, 2992]*1.5486+0.1699
            hidden_states[:, :, 320] = hidden_states[:, :, 320]*1.6443+0.5721
            hidden_states[:, :, 3381] = hidden_states[:, :, 3381]*1.7557+0.301
            hidden_states[:, :, 2584] = hidden_states[:, :, 2584]*1.2938-0.648
            return (hidden_states,) + tuple(other_outputs)
        return zero_column_hook_1
    hook_1 = create_hook_1()
    target_layer_1 = model.model.layers[22]
    target_layer_1.register_forward_hook(hook_1)    
    model.config.num_hidden_layers = len(model.model.layers)
    model.to('cuda:0')
    torch.autograd.set_detect_anomaly(False)
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    print('model ppl is:',dataset_ppl)
    
        
if __name__ == "__main__":
    pcc.utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    slicing_args = slicing_arg_parser()
    process_slicing_args(slicing_args)
    slicing_main(slicing_args)
    
