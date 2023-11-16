#  -*- coding: utf-8 -*-
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class HotelReviewDataset(Dataset):
    def __init__(self, data_filepath):
        self.raw_data = self.read_data(data_filepath)

    @staticmethod
    def read_data(filepath):
        with open(filepath, encoding="utf-8") as f:
            content = json.load(f)
        return content

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, item):
        example = self.raw_data[item]
        return {"review": example["text"], "label": example["polarity"]}


class HotelReviewDataCollator(object):
    def __init__(
        self, max_seq_len=None, pretrained_model_path=None, pad_to_multiple_of=None
    ):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
        self.max_seq_len = (
            self.tokenizer.model_max_length if max_seq_len is None else max_seq_len
        )
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label2id = {"pos": 0, "neg": 1}

    def __call__(self, examples):
        # print(examples)
        reviews = [item["review"] for item in examples]
        tokenized_outputs = self.tokenizer(
            text=reviews,
            max_length=self.max_seq_len,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_token_type_ids=False,
            padding="max_length",
            return_tensors="pt",
        )
        # self.tokenizer.pad()
        if "label" in examples[0]:
            labels = [self.label2id.get(item["label"]) for item in examples]
            tokenized_outputs.update(
                {"labels": torch.as_tensor(labels, dtype=torch.long)}
            )
        # print(tokenized_outputs)
        return tokenized_outputs
