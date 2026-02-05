import logging

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase


def get_dataset(name: str) -> datasets.DatasetDict:
   
    logging.info(f"Loading dataset: {name}")

    ds_properties = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "config_name": "allenai--c4",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "cols_to_remove": ['url', 'timestamp'],
        },
        "alpaca": {"path": "tatsu-lab/alpaca", "cols_to_remove": ['input', 'output', 'instruction']},
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")

    properties = ds_properties[name]
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files")
    )

    if "cols_to_remove" in properties:
        ds = ds.remove_columns(properties["cols_to_remove"])

    if name == "alpaca":
        ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        temp_ds = ds.pop("test")
        temp_ds = temp_ds.train_test_split(test_size=0.5, seed=42)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    logging.info("Loading dataset done")
    return ds


def prepare_test_dataloader(
    dataset: datasets.Dataset, tokenizer: PreTrainedTokenizerBase, seqlen: int = 2048, batch_size: int = 1
) -> DataLoader[dict[str, torch.Tensor]]:

    logging.info(f"Preparing test dataloader")

    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):

            tokenized_ds = tokenizer("\n\n".join(ds['text']), return_tensors='pt')
            nsamples = tokenized_ds.input_ids.numel() // seqlen

            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    loader = DataLoader(test_ds, batch_size=batch_size)
    logging.info(f"Preparing test dataloader done")
    return loader


def prepare_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed=42,
) -> DataLoader[dict[str, torch.Tensor]]:
    logging.info(f"Preparing dataloader")

    if not varied_seqlen and not nsamples:
        logging.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        data_list = ds[data_name]
        new_data_list = []

        torch.manual_seed(seed)
        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,)).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    ds.set_transform(tokenize)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds))[:nsamples])

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    logging.info(f"Preparing dataloader done")
    return loader
