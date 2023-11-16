#  -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = f"1"

import random
import warnings
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup

# transformers.logging.set_verbosity_error()

"""
如果出现长度超出最大长度被截断的情况，就会出现如下警告信息，如果不想看到警告，可以设置 transformers.logging.set_verbosity_error()
Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. 
sequence pairs with the 'longest_first' truncation strategy. So the returned list will always be empty even if some tokens have been removed.
"""

from build_dataset import ChatGLMDataset, ChatGLMDataCollator
from chatglm_model import ChatGLMForConditionalGeneration

warnings.filterwarnings("ignore")


base_args = Namespace(
    test_data_filepath="test.json",
    train_data_filepath="/home/lis/algProjects/finetuneChatGLM/train_data.json",
    # train_data_filepath="/home/lis/algProjects/finetuneChatGLM/training_data_nlp.json",
    valid_data_filepath="/home/lis/algProjects/finetuneChatGLM/valid_data.json",
    pretrained_model_path="/date/pretrained_models/chatglm2-6b/",
    checkpoint_filepath="chatglm2_test_lora.pth",
    # max_seq_len=256,
    max_seq_len=256,
    train_batch_size=16,
    eval_batch_size=16,
    # batch_size=4,
    batch_size=2,
    learning_rate=3e-4,
    warmup_proportion=0.1,
    epochs=6,
    gradient_accumulation_steps=1,
    seed=666,
    device=torch.device(f"cuda" if torch.cuda.is_available() else "cpu"),
)


def get_args():
    parser = ArgumentParser()
    # GPU配置参数
    parser.add_argument(
        "--gpu_devices", type=int, nargs="+", default=[0, 1, 2], help=""
    )  # 改变量传递的是一个列表参数
    args = parser.parse_args()
    for key, value in base_args.__dict__.items():
        setattr(args, key, value)
    return args


def get_optimizer(model, learning_rate):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # param_optimizer = list(model.named_parameters())
    param_optimizer = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': 0.01,
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, no_deprecation_warning=True)
    return optimizer


def get_model(args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['query_key_value', "dense_h_to_4h", "dense_4h_to_h"],
        fan_in_fan_out=False
    )
    model = ChatGLMForConditionalGeneration.from_pretrained(
        args.pretrained_model_path,
        device_map="auto",
        torch_dtype=torch.half
    )
    # 对embedding层进行权重更新，会增加显存占用，暂时先不开启。
    model.enable_input_require_grads()
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.to(args.device)
    return model


def train_and_eval(args, model, optimizer, train_loader, valid_loader, loss_or_metric="loss"):
    best_eval_f1 = 0
    best_eval_loss = np.inf
    scaler = torch.cuda.amp.GradScaler()
    # 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
    num_training_steps = int(
        len(train_loader) * args.epochs / args.gradient_accumulation_steps
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_training_steps),
        num_training_steps=num_training_steps,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for step, batch_dict in enumerate(tqdm(train_loader, desc="进行训练"), start=1):
            batch_dict = {key: value.to(args.device) for key, value in batch_dict.items()}
            with torch.autocast("cuda"):
                loss = model(**batch_dict).loss
                print(loss)
                train_loss += loss.item()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # backward
                # Accumulates scaled gradients.
                scaler.scale(loss).backward()
                # weights update
                if step % args.gradient_accumulation_steps == 0:
                    # may unscale_ here if desired
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
        train_loss /= len(train_loader.sampler)
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_dict in tqdm(valid_loader, desc="进行评估"):
                batch_dict = {key: value.to(args.device) for key, value in batch_dict.items()}
                with torch.cuda.amp.autocast():
                    loss = model(**batch_dict).loss
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    valid_loss += loss.item()
            valid_loss /= len(valid_loader.sampler)

            eval_micro_f1 = 0
            """Any methods that download data should be isolated to the master process.
            Any methods that perform file I/O should be isolated to the master process."""
            print(
                f"epoch:{epoch} ||train_loss:{train_loss} ||valid_loss:{valid_loss}"
            )
            if loss_or_metric == "loss":
                if valid_loss < best_eval_loss:
                    use_torch = False
                    if use_torch:
                        saved_params = {
                            k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
                        }
                        torch.save(saved_params, args.checkpoint_filepath)
                    else:
                        model.save_pretrained("./lora_model")

                    # torch.save(model.module.state_dict(), args.checkpoint_filepath)
                    best_eval_loss = valid_loss
            elif loss_or_metric == "metric":
                if eval_micro_f1 > best_eval_f1:
                    torch.save(model.state_dict(), args.checkpoint_filepath)
                    best_eval_f1 = eval_micro_f1
            else:
                raise ValueError("在验证集上选取保存模型的方式有误。")
        model.train()


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = get_args()
    set_seed(args.seed)
    data_collator = ChatGLMDataCollator(args.max_seq_len, args.pretrained_model_path)
    train_dataset = ChatGLMDataset(args.train_data_filepath)
    valid_dataset = ChatGLMDataset(args.valid_data_filepath)

    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=4,  # 单卡训练时不选0，可以加快数据加载的速度。
        collate_fn=data_collator,
        drop_last=False,  # drop_last默认为False，也就是会使用已有的样本进行填充，保证每个GPU上分配的数据都是完整的batch_size
    )

    valid_loader = DataLoader(
        valid_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,  # 看下这个参数是否有必要
        collate_fn=data_collator,
    )
    model = get_model(args)
    optimizer = get_optimizer(model, learning_rate=args.learning_rate)
    train_and_eval(
        args, model, optimizer, train_loader, valid_loader
    )


if __name__ == "__main__":
    main()
