"""
Load and tokenize SST-2 dataset using HuggingFace Datasets and Tokenizers.
"""

from datasets import load_dataset
from transformers import BertTokenizer
import torch

def load_sst2(max_len=64):
    dataset = load_dataset("glue", "sst2")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        tokens = tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=max_len)
        return {"input_ids": tokens["input_ids"], "label": example["label"]}

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch", columns=["input_ids", "label"])

    train_data = dataset["train"]
    test_data = dataset["validation"]

    return train_data, test_data
