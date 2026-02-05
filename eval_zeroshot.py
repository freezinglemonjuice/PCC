import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import copy
from tqdm import tqdm
import fire
import os
from typing import List
from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils.model_utils import get_llm
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.block_remove import block_remove
def eval(
        model_name: str = 'meta-llama/Meta-Llama-3-8B',
        removal_list: List[int] = [],
        save_results: bool = True,
        result_folder: str = 'results',
        result_file: str = 'eval.txt',
        device: int = 0,
        eval_zeroshot: bool = False
    ):
    model = get_llm(model_name)
    print(f"Loaded Model: {model.name}")
    model.eval()
    original_removal_list = copy.deepcopy(removal_list)
    removal_list.sort()
    del model
    torch.cuda.empty_cache()

    if eval_zeroshot:
        print(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        tasks = ['arc_challenge','arc_easy','boolq','hellaswag','piqa','winogrande','wsc','mnli','anli']
        results = eval_zero_shot('model_path', copy.deepcopy(removal_list), tasks, parallelize=parallelize)
        results = results['results']
        for task in tasks:
            print(f"{task}: {results[task]}")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_path = os.path.join(result_folder, result_file)
if __name__ == "__main__":
    fire.Fire(eval)