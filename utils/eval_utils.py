import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn

from utils.data_utils import *
import fnmatch


@torch.no_grad()
def load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2', testloader=None, tokenizer=None):
    print(f"Evaluating on {dataset}")

    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model.name)

        _, testloader = get_loaders(
            dataset, seed=42, seqlen=model.seqlen, tokenizer=tokenizer 
        )
        print(f"Dataset Loaded.")

    with torch.no_grad():
        ppl_test = eval_ppl(model, testloader, 1, device)
    return ppl_test 

@torch.no_grad()
def eval_ppl(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids

    nsamples = testenc.numel() // model.seqlen

    nlls = []
    print(f"nsamples {nsamples}")

    for i in tqdm(range(0,nsamples,bs)):

        j = min(i+bs, nsamples)

        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))


    return ppl.item()

torch.no_grad()
def eval_zero_shot(model_name, removal_list, task_list=['arc_challenge','arc_easy'], 
        num_fewshot=0, parallelize=False):
    
    from lm_evaluation_harness.lm_eval import tasks, evaluator, utils
    task_manager = tasks.TaskManager(include_path='lm_evaluation_harness/lm_eval/tasks')
 
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
    task_missing = [
        task
        for task in task_list
        if task not in task_names and "*" not in task
        ]  
    
    
    model_args = f"pretrained={model_name},"
    if parallelize:
        model_args = f"pretrained={model_name},parallelize=True"

    if len(removal_list)>0:
        remove = True
    else:
        remove = False

    results = evaluator.simple_evaluate(
        model='hf',
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size='auto',
        max_batch_size=None,
        device='cuda:0',
        use_cache=None,
        limit=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        gen_kwargs=None,
        task_manager=task_manager,
        remove = remove,
        removal_list = removal_list,
    )

    return results 