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
        #################layer9
        model.model.layers[9].self_attn.q_proj.weight.data=model.model.layers[13].self_attn.q_proj.weight.data
        model.model.layers[9].self_attn.k_proj.weight.data=model.model.layers[13].self_attn.k_proj.weight.data
        model.model.layers[9].self_attn.v_proj.weight.data=model.model.layers[13].self_attn.v_proj.weight.data
        model.model.layers[9].self_attn.o_proj.weight.data=model.model.layers[13].self_attn.o_proj.weight.data
        model.model.layers[9].mlp.gate_proj.weight.data=model.model.layers[13].mlp.gate_proj.weight.data
        model.model.layers[9].mlp.up_proj.weight.data=model.model.layers[13].mlp.up_proj.weight.data
        model.model.layers[9].mlp.down_proj.weight.data=model.model.layers[13].mlp.down_proj.weight.data
        model.model.layers[9].input_layernorm.weight.data=model.model.layers[13].input_layernorm.weight.data
        model.model.layers[9].post_attention_layernorm.weight.data=model.model.layers[13].post_attention_layernorm.weight.data
        #################layer10
        model.model.layers[10].self_attn.q_proj.weight.data=model.model.layers[14].self_attn.q_proj.weight.data
        model.model.layers[10].self_attn.k_proj.weight.data=model.model.layers[14].self_attn.k_proj.weight.data
        model.model.layers[10].self_attn.v_proj.weight.data=model.model.layers[14].self_attn.v_proj.weight.data
        model.model.layers[10].self_attn.o_proj.weight.data=model.model.layers[14].self_attn.o_proj.weight.data
        model.model.layers[10].mlp.gate_proj.weight.data=model.model.layers[14].mlp.gate_proj.weight.data
        model.model.layers[10].mlp.up_proj.weight.data=model.model.layers[14].mlp.up_proj.weight.data
        model.model.layers[10].mlp.down_proj.weight.data=model.model.layers[14].mlp.down_proj.weight.data
        model.model.layers[10].input_layernorm.weight.data=model.model.layers[14].input_layernorm.weight.data
        model.model.layers[10].post_attention_layernorm.weight.data=model.model.layers[14].post_attention_layernorm.weight.data
        #################layer11
        model.model.layers[11].self_attn.q_proj.weight.data=model.model.layers[15].self_attn.q_proj.weight.data
        model.model.layers[11].self_attn.k_proj.weight.data=model.model.layers[15].self_attn.k_proj.weight.data
        model.model.layers[11].self_attn.v_proj.weight.data=model.model.layers[15].self_attn.v_proj.weight.data
        model.model.layers[11].self_attn.o_proj.weight.data=model.model.layers[15].self_attn.o_proj.weight.data
        model.model.layers[11].mlp.gate_proj.weight.data=model.model.layers[15].mlp.gate_proj.weight.data
        model.model.layers[11].mlp.up_proj.weight.data=model.model.layers[15].mlp.up_proj.weight.data
        model.model.layers[11].mlp.down_proj.weight.data=model.model.layers[15].mlp.down_proj.weight.data
        model.model.layers[11].input_layernorm.weight.data=model.model.layers[15].input_layernorm.weight.data
        model.model.layers[11].post_attention_layernorm.weight.data=model.model.layers[15].post_attention_layernorm.weight.data
        #################layer12
        model.model.layers[12].self_attn.q_proj.weight.data=model.model.layers[16].self_attn.q_proj.weight.data
        model.model.layers[12].self_attn.k_proj.weight.data=model.model.layers[16].self_attn.k_proj.weight.data
        model.model.layers[12].self_attn.v_proj.weight.data=model.model.layers[16].self_attn.v_proj.weight.data
        model.model.layers[12].self_attn.o_proj.weight.data=model.model.layers[16].self_attn.o_proj.weight.data
        model.model.layers[12].mlp.gate_proj.weight.data=model.model.layers[16].mlp.gate_proj.weight.data
        model.model.layers[12].mlp.up_proj.weight.data=model.model.layers[16].mlp.up_proj.weight.data
        model.model.layers[12].mlp.down_proj.weight.data=model.model.layers[16].mlp.down_proj.weight.data
        model.model.layers[12].input_layernorm.weight.data=model.model.layers[16].input_layernorm.weight.data
        model.model.layers[12].post_attention_layernorm.weight.data=model.model.layers[16].post_attention_layernorm.weight.data
        #################layer13
        model.model.layers[13].self_attn.q_proj.weight.data=model.model.layers[17].self_attn.q_proj.weight.data
        model.model.layers[13].self_attn.k_proj.weight.data=model.model.layers[17].self_attn.k_proj.weight.data
        model.model.layers[13].self_attn.v_proj.weight.data=model.model.layers[17].self_attn.v_proj.weight.data
        model.model.layers[13].self_attn.o_proj.weight.data=model.model.layers[17].self_attn.o_proj.weight.data
        model.model.layers[13].mlp.gate_proj.weight.data=model.model.layers[17].mlp.gate_proj.weight.data
        model.model.layers[13].mlp.up_proj.weight.data=model.model.layers[17].mlp.up_proj.weight.data
        model.model.layers[13].mlp.down_proj.weight.data=model.model.layers[17].mlp.down_proj.weight.data
        model.model.layers[13].input_layernorm.weight.data=model.model.layers[17].input_layernorm.weight.data
        model.model.layers[13].post_attention_layernorm.weight.data=model.model.layers[17].post_attention_layernorm.weight.data
        #################layer14
        model.model.layers[14].self_attn.q_proj.weight.data=model.model.layers[18].self_attn.q_proj.weight.data
        model.model.layers[14].self_attn.k_proj.weight.data=model.model.layers[18].self_attn.k_proj.weight.data
        model.model.layers[14].self_attn.v_proj.weight.data=model.model.layers[18].self_attn.v_proj.weight.data
        model.model.layers[14].self_attn.o_proj.weight.data=model.model.layers[18].self_attn.o_proj.weight.data
        model.model.layers[14].mlp.gate_proj.weight.data=model.model.layers[18].mlp.gate_proj.weight.data
        model.model.layers[14].mlp.up_proj.weight.data=model.model.layers[18].mlp.up_proj.weight.data
        model.model.layers[14].mlp.down_proj.weight.data=model.model.layers[18].mlp.down_proj.weight.data
        model.model.layers[14].input_layernorm.weight.data=model.model.layers[18].input_layernorm.weight.data
        model.model.layers[14].post_attention_layernorm.weight.data=model.model.layers[18].post_attention_layernorm.weight.data
        #################layer15
        model.model.layers[15].self_attn.q_proj.weight.data=model.model.layers[19].self_attn.q_proj.weight.data
        model.model.layers[15].self_attn.k_proj.weight.data=model.model.layers[19].self_attn.k_proj.weight.data
        model.model.layers[15].self_attn.v_proj.weight.data=model.model.layers[19].self_attn.v_proj.weight.data
        model.model.layers[15].self_attn.o_proj.weight.data=model.model.layers[19].self_attn.o_proj.weight.data
        model.model.layers[15].mlp.gate_proj.weight.data=model.model.layers[19].mlp.gate_proj.weight.data
        model.model.layers[15].mlp.up_proj.weight.data=model.model.layers[19].mlp.up_proj.weight.data
        model.model.layers[15].mlp.down_proj.weight.data=model.model.layers[19].mlp.down_proj.weight.data
        model.model.layers[15].input_layernorm.weight.data=model.model.layers[19].input_layernorm.weight.data
        model.model.layers[15].post_attention_layernorm.weight.data=model.model.layers[19].post_attention_layernorm.weight.data
        #################layer16
        model.model.layers[16].self_attn.q_proj.weight.data=model.model.layers[20].self_attn.q_proj.weight.data
        model.model.layers[16].self_attn.k_proj.weight.data=model.model.layers[20].self_attn.k_proj.weight.data
        model.model.layers[16].self_attn.v_proj.weight.data=model.model.layers[20].self_attn.v_proj.weight.data
        model.model.layers[16].self_attn.o_proj.weight.data=model.model.layers[20].self_attn.o_proj.weight.data
        model.model.layers[16].mlp.gate_proj.weight.data=model.model.layers[20].mlp.gate_proj.weight.data
        model.model.layers[16].mlp.up_proj.weight.data=model.model.layers[20].mlp.up_proj.weight.data
        model.model.layers[16].mlp.down_proj.weight.data=model.model.layers[20].mlp.down_proj.weight.data
        model.model.layers[16].input_layernorm.weight.data=model.model.layers[20].input_layernorm.weight.data
        model.model.layers[16].post_attention_layernorm.weight.data=model.model.layers[20].post_attention_layernorm.weight.data
        ##################layer17
        model.model.layers[17].self_attn.q_proj.weight.data=model.model.layers[21].self_attn.q_proj.weight.data
        model.model.layers[17].self_attn.k_proj.weight.data=model.model.layers[21].self_attn.k_proj.weight.data
        model.model.layers[17].self_attn.v_proj.weight.data=model.model.layers[21].self_attn.v_proj.weight.data
        model.model.layers[17].self_attn.o_proj.weight.data=model.model.layers[21].self_attn.o_proj.weight.data
        model.model.layers[17].mlp.gate_proj.weight.data=model.model.layers[21].mlp.gate_proj.weight.data
        model.model.layers[17].mlp.up_proj.weight.data=model.model.layers[21].mlp.up_proj.weight.data
        model.model.layers[17].mlp.down_proj.weight.data=model.model.layers[21].mlp.down_proj.weight.data
        model.model.layers[17].input_layernorm.weight.data=model.model.layers[21].input_layernorm.weight.data
        model.model.layers[17].post_attention_layernorm.weight.data=model.model.layers[21].post_attention_layernorm.weight.data
        #################layer18
        model.model.layers[18].self_attn.q_proj.weight.data=model.model.layers[22].self_attn.q_proj.weight.data
        model.model.layers[18].self_attn.k_proj.weight.data=model.model.layers[22].self_attn.k_proj.weight.data
        model.model.layers[18].self_attn.v_proj.weight.data=model.model.layers[22].self_attn.v_proj.weight.data
        model.model.layers[18].self_attn.o_proj.weight.data=model.model.layers[22].self_attn.o_proj.weight.data
        model.model.layers[18].mlp.gate_proj.weight.data=model.model.layers[22].mlp.gate_proj.weight.data
        model.model.layers[18].mlp.up_proj.weight.data=model.model.layers[22].mlp.up_proj.weight.data
        model.model.layers[18].mlp.down_proj.weight.data=model.model.layers[22].mlp.down_proj.weight.data
        model.model.layers[18].input_layernorm.weight.data=model.model.layers[22].input_layernorm.weight.data
        model.model.layers[18].post_attention_layernorm.weight.data=model.model.layers[22].post_attention_layernorm.weight.data
        #################layer19
        model.model.layers[19].self_attn.q_proj.weight.data=model.model.layers[24].self_attn.q_proj.weight.data
        model.model.layers[19].self_attn.k_proj.weight.data=model.model.layers[24].self_attn.k_proj.weight.data
        model.model.layers[19].self_attn.v_proj.weight.data=model.model.layers[24].self_attn.v_proj.weight.data
        model.model.layers[19].self_attn.o_proj.weight.data=model.model.layers[24].self_attn.o_proj.weight.data
        model.model.layers[19].mlp.gate_proj.weight.data=model.model.layers[24].mlp.gate_proj.weight.data
        model.model.layers[19].mlp.up_proj.weight.data=model.model.layers[24].mlp.up_proj.weight.data
        model.model.layers[19].mlp.down_proj.weight.data=model.model.layers[24].mlp.down_proj.weight.data
        model.model.layers[19].input_layernorm.weight.data=model.model.layers[24].input_layernorm.weight.data
        model.model.layers[19].post_attention_layernorm.weight.data=model.model.layers[24].post_attention_layernorm.weight.data
        #################layer20
        model.model.layers[20].self_attn.q_proj.weight.data=model.model.layers[27].self_attn.q_proj.weight.data
        model.model.layers[20].self_attn.k_proj.weight.data=model.model.layers[27].self_attn.k_proj.weight.data
        model.model.layers[20].self_attn.v_proj.weight.data=model.model.layers[27].self_attn.v_proj.weight.data
        model.model.layers[20].self_attn.o_proj.weight.data=model.model.layers[27].self_attn.o_proj.weight.data
        model.model.layers[20].mlp.gate_proj.weight.data=model.model.layers[27].mlp.gate_proj.weight.data
        model.model.layers[20].mlp.up_proj.weight.data=model.model.layers[27].mlp.up_proj.weight.data
        model.model.layers[20].mlp.down_proj.weight.data=model.model.layers[27].mlp.down_proj.weight.data
        model.model.layers[20].input_layernorm.weight.data=model.model.layers[27].input_layernorm.weight.data
        model.model.layers[20].post_attention_layernorm.weight.data=model.model.layers[27].post_attention_layernorm.weight.data
        #################layer21
        model.model.layers[21].self_attn.q_proj.weight.data=model.model.layers[28].self_attn.q_proj.weight.data
        model.model.layers[21].self_attn.k_proj.weight.data=model.model.layers[28].self_attn.k_proj.weight.data
        model.model.layers[21].self_attn.v_proj.weight.data=model.model.layers[28].self_attn.v_proj.weight.data
        model.model.layers[21].self_attn.o_proj.weight.data=model.model.layers[28].self_attn.o_proj.weight.data
        model.model.layers[21].mlp.gate_proj.weight.data=model.model.layers[28].mlp.gate_proj.weight.data
        model.model.layers[21].mlp.up_proj.weight.data=model.model.layers[28].mlp.up_proj.weight.data
        model.model.layers[21].mlp.down_proj.weight.data=model.model.layers[28].mlp.down_proj.weight.data
        model.model.layers[21].input_layernorm.weight.data=model.model.layers[28].input_layernorm.weight.data
        model.model.layers[21].post_attention_layernorm.weight.data=model.model.layers[28].post_attention_layernorm.weight.data
        #################layer22
        model.model.layers[22].self_attn.q_proj.weight.data=model.model.layers[29].self_attn.q_proj.weight.data
        model.model.layers[22].self_attn.k_proj.weight.data=model.model.layers[29].self_attn.k_proj.weight.data
        model.model.layers[22].self_attn.v_proj.weight.data=model.model.layers[29].self_attn.v_proj.weight.data
        model.model.layers[22].self_attn.o_proj.weight.data=model.model.layers[29].self_attn.o_proj.weight.data
        model.model.layers[22].mlp.gate_proj.weight.data=model.model.layers[29].mlp.gate_proj.weight.data
        model.model.layers[22].mlp.up_proj.weight.data=model.model.layers[29].mlp.up_proj.weight.data
        model.model.layers[22].mlp.down_proj.weight.data=model.model.layers[29].mlp.down_proj.weight.data
        model.model.layers[22].input_layernorm.weight.data=model.model.layers[29].input_layernorm.weight.data
        model.model.layers[22].post_attention_layernorm.weight.data=model.model.layers[29].post_attention_layernorm.weight.data
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
            hidden_states[:, :, 2932] = hidden_states[:, :, 2932]*1.2695+0.1022
            hidden_states[:, :, 1189] = hidden_states[:, :, 1189]*1.3478+0.0133
            hidden_states[:, :, 1298] = hidden_states[:, :, 1298]*1.1706-0.0298
            hidden_states[:, :, 291] = hidden_states[:, :, 291]*1.3281+0.0503
            hidden_states[:, :, 3266] = hidden_states[:, :, 3266]*1.2505-0.0458
            hidden_states[:, :, 1162] = hidden_states[:, :, 1162]*1.3009+0.001
            hidden_states[:, :, 133] = hidden_states[:, :, 133]*1.2804+0.0031
            hidden_states[:, :, 2742] = hidden_states[:, :, 2742]*1.4084-0.1651
            hidden_states[:, :, 1039] = hidden_states[:, :, 1039]*1.2874+0.0026
            hidden_states[:, :, 3231] = hidden_states[:, :, 3231]*1.2358-0.0149
            hidden_states[:, :, 2720] = hidden_states[:, :, 2720]*1.358-0.0701
            hidden_states[:, :, 761] = hidden_states[:, :, 761]*1.4782+0.0438
            hidden_states[:, :, 1815] = hidden_states[:, :, 1815]*1.2736-0.0977
            hidden_states[:, :, 2977] = hidden_states[:, :, 2977]*1.239+0.0672
            hidden_states[:, :, 3516] = hidden_states[:, :, 3516]*1.273-0.0397
            hidden_states[:, :, 2692] = hidden_states[:, :, 2692]*1.2824+0.0315
            hidden_states[:, :, 709] = hidden_states[:, :, 709]*1.185+0.0405
            hidden_states[:, :, 3881] = hidden_states[:, :, 3881]*1.2042-0.0846
            return (hidden_states,) + tuple(other_outputs)
        return zero_column_hook_1
    hook_1 = create_hook_1()
    target_layer_1 = model.model.layers[8]
    target_layer_1.register_forward_hook(hook_1)

    
    def create_hook_2():
        def zero_column_hook_2(module, input, output):
            hidden_states, *other_outputs = output
            hidden_states[:, :, 2352] = hidden_states[:, :, 2352]*1.2241-0.3702
            hidden_states[:, :, 4055] = hidden_states[:, :, 4055]*0.9199-0.2198
            hidden_states[:, :, 2209] = hidden_states[:, :, 2209]*1.1811+0.018
            hidden_states[:, :, 709] = hidden_states[:, :, 709]*1.0417-0.1218
            hidden_states[:, :, 1951] = hidden_states[:, :, 1951]*1.1561+0.0138
            hidden_states[:, :, 1736] = hidden_states[:, :, 1736]*1.1322-0.0669
            hidden_states[:, :, 2613] = hidden_states[:, :, 2613]*1.1196+0.0341
            hidden_states[:, :, 3965] = hidden_states[:, :, 3965]*1.1907+0.0274
            hidden_states[:, :, 1658] = hidden_states[:, :, 1658]*1.1103+0.0116
            hidden_states[:, :, 1059] = hidden_states[:, :, 1059]*1.0425+0.08
            hidden_states[:, :, 357] = hidden_states[:, :, 357]*1.135+0.0397
            hidden_states[:, :, 2252] = hidden_states[:, :, 2252]*1.1596+0.0167
            return (hidden_states,) + tuple(other_outputs)
        return zero_column_hook_2
    hook_2 = create_hook_2()
    target_layer_2 = model.model.layers[18]
    target_layer_2.register_forward_hook(hook_2)

    
    def create_hook_3():
        def zero_column_hook_3(module, input, output):
            hidden_states, *other_outputs = output
            hidden_states[:, :, 2352] = hidden_states[:, :, 2352]*1.1962-1.5163
            hidden_states[:, :, 2252] = hidden_states[:, :, 2252]*1.8405+0.389
            hidden_states[:, :, 1384] = hidden_states[:, :, 1384]*1.0053+0.3459
            hidden_states[:, :, 1658] = hidden_states[:, :, 1658]*1.1721+0.1263
            hidden_states[:, :, 349] = hidden_states[:, :, 349]*1.2153+0.1875
            hidden_states[:, :, 229] = hidden_states[:, :, 229]*1.1938-0.2105
            hidden_states[:, :, 3585] = hidden_states[:, :, 3585]*1.1462+0.0775
            hidden_states[:, :, 2613] = hidden_states[:, :, 2613]*1.1672+0.0284
            return (hidden_states,) + tuple(other_outputs)
        return zero_column_hook_3
    hook_3 = create_hook_3()
    target_layer_3 = model.model.layers[19]
    target_layer_3.register_forward_hook(hook_3)
    
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
    
