# PCC: Pivotal Channel Compensation for Layer Pruning of Large Language Models

## To install the pcc package, we recommed
```python
pip install -e .[experiment]
```

## Running PCC
### Calculate the PPL on LLaMA3-8B Model
#### PCC with Consecutive-layer Pruning (LLM-Streamline) 
```python
python run_llama3_8b_llm.py \
      --model meta-llama/Meta-Llama-3-8B \
      --device cuda:0 \
      --no-wandb
```
#### PCC with Consecutive-layer Pruning (ShortGPT)
```python
python run_llama3_8b_Short.py \
      --model meta-llama/Meta-Llama-3-8B \
      --device cuda:0 \
      --no-wandb
```
#### PCC with Cross-layer Pruning (SLEB)
```python
python run_llama3_8b_SLEB.py \
      --model meta-llama/Meta-Llama-3-8B \
      --device cuda:0 \
      --no-wandb
```

### Calculate the Zero-shot tasks on LLaMA3-8B Model
#### Step 1
Save the model weights after layer pruning.
#### Step 2
Fill in the model weights address in the specified position in eval_zeroshot.py.
#### Step 3 
Configure the lm-evaluation-harness environments
#### Step 4
Replace evaluator.py in lm-evaluation-harness with evaluator_8b_SLEB, evaluator_8b_llm, evaluator_8b_Short respectively.
#### Step 5
```python
python -m eval_zeroshot.py \ 
       --model_name meta-llama/Meta-Llama-3-8B \
       --removal_list '' \
       --eval_zeroshot True
```






