#  -*- coding: utf-8 -*-
import json
import random
import warnings

import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer

from chatglm2_tokenizer import ChatGLMTokenizer

warnings.filterwarnings("ignore")


class ChatGLMDataset(Dataset):
    def __init__(self, filepath):
        self.raw_data = self.convert_and_clean_data(self.read_data(filepath))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, item):
        example = self.raw_data[item]
        return {"context": example["context"], "target": example["target"]}

    def convert_and_clean_data(self, raw_data: list):
        raw_data = map(self.format_example, raw_data)
        raw_data = list(
            filter(
                lambda example: example["target"] is not None
                                and example["context"] is not None,
                raw_data,
            )
        )
        return raw_data

    @staticmethod
    def read_data(filepath):
        """
            {
          "instruction": "说出初次旅行的人应该考虑的三件事。",
          "input": "",
          "output": "初次旅行的人应该考虑旅行的费用、旅行的长度以及他们希望在旅途中参加的活动。"
        }

        {
        "instruction": "计算给定数据集中增加或减少的百分比。",
        "input": "五年前，公司有 10,000 名员工，现在公司有 15,000 名员工。",
        "output": "增加 50%"
        }
        """
        with open(filepath, encoding="utf-8") as f:
            content = json.load(f)
        random.shuffle(content)
        return content

    @staticmethod
    def format_example(example: dict, use_src=False) -> dict:
        """原始数据格式转换为新的数据格式"""
        if use_src:
            context = f"Instruction: {example['instruction']}\n"
            if example.get("input"):
                context += f"Input: {example['input']}\n"
            context += "Answer: "
            target = example["output"]
            example["context"] = context
            example["target"] = target
        else:
            context = f"问：{example['instruction']}\n\n"
            if example.get("input"):
                context += f"{example['input']}\n\n"
            context += "答："
            target = example["output"]
            example["context"] = context
            example["target"] = target
        return example


class ChatGLMDataCollator(object):
    def __init__(
            self,
            max_seq_len=512,
            pretrained_model_path=None,
            use_local=False
    ):
        if use_local:
            self.tokenizer = ChatGLMTokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path, trust_remote_code=True, revision="dev"
            )

        self.max_seq_len = max_seq_len

    def __call__(self, examples, use_prompt_loss=False):
        context = [example["context"] for example in examples]
        target = [example["target"] for example in examples]

        tokenized_outputs = self.tokenizer(
            context,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            text_pair=target,
            text_target=target,
            return_tensors="pt",
        )
        input_ids = tokenized_outputs["input_ids"]
        if use_prompt_loss:
            labels = tokenized_outputs["labels"]
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels = torch.cat(
                [
                    labels[:, 1:],
                    input_ids[:, -1].view(-1, 1)
                ],
                dim=-1
            )
            labels[labels == 64790] = -100
            labels[labels == 64792] = -100
            tokenized_outputs["labels"] = labels
        else:
            labels = torch.where(input_ids != self.tokenizer.pad_token_id, input_ids, -100)
            tokenized_outputs["labels"] = torch.cat(
                (
                    labels[:, :-1],
                    torch.as_tensor(
                        # [self.tokenizer.get_command("eop")] * input_ids.size(0)
                        [self.tokenizer.pad_token_id] * input_ids.size(0)
                    ).view(-1, 1)
                ),
                dim=-1,
            )
        return tokenized_outputs


if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(
        valid_data_filepath="/home/lis/algProjects/finetuneChatGLM/train_data.json",
        max_seq_len=128,
        pretrained_model_path="/home/lis/algProjects/pretrained_models/chatglm2-6b/",
        # pretrained_model_path="/date/pretrained_models/ChatGLM/",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    data_collator = ChatGLMDataCollator(
        max_seq_len=args.max_seq_len, pretrained_model_path=args.pretrained_model_path
    )

    chatglm_dataset = ChatGLMDataset(args.valid_data_filepath)
    #
    data_loader = DataLoader(chatglm_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)
    batch = next(iter(data_loader))
    # print(data_collator.tokenizer.pad_token_id, data_collator.tokenizer.eos_token_id)
    #
    print(batch["input_ids"])
    print(batch["labels"])
