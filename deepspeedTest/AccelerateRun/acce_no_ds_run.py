#  -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from accelerate import Accelerator
from accelerate.utils import set_seed
import evaluate
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace, ArgumentParser

import sys

sys.path.append('/home/lis/algProjects/hotelReviewCls/')
from build_dataset import HotelReviewDataset, HotelReviewDataCollator


def get_optimizer(model, learning_rate):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    # param_optimizer = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def get_metrics(pred, ref):
    pred = torch.cat(pred, dim=-1)
    ref = torch.cat(ref, dim=-1)
    print("验证集样本数量：", pred.size(0))
    acc_metric = evaluate.load("accuracy")
    f_metric = evaluate.load("f1")
    r_metric = evaluate.load("recall")
    p_metric = evaluate.load("precision")
    result = dict()
    result.update(acc_metric.compute(predictions=pred, references=ref))
    result.update(f_metric.compute(predictions=pred, references=ref, average="macro"))
    result.update(p_metric.compute(predictions=pred, references=ref, average="macro"))
    result.update(r_metric.compute(predictions=pred, references=ref, average="macro"))
    return result


def main(args):
    set_seed(args.seed)
    # Initialize accelerator
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.use_cpu,
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            project_dir=args.logging_dir,
            device_placement=True,
            split_batches=False,
        )
    else:
        accelerator = Accelerator(
            cpu=args.use_cpu,
            mixed_precision=args.mixed_precision,
            device_placement=True,  # 不用再对model以及tensor进行指定了。
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            split_batches=False,
        )

    print("word_size是多少", dist.get_world_size())
    # 选择是每个epoch介绍保存模型，还是每隔指定的step保存模型。
    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, args.__dict__)

    with accelerator.main_process_first():
        train_dataset = HotelReviewDataset(args.train_data_filepath)
        eval_dataset = HotelReviewDataset(args.valid_data_filepath)

    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16  # multiple 倍数,填充序列长度为16的倍数，
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None
    data_collator = HotelReviewDataCollator(
        max_seq_len=args.max_seq_len,
        pretrained_model_path=args.pretrained_model_path,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    valid_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = BertForSequenceClassification.from_pretrained(args.pretrained_model_path)
    # model = model.to(accelerator.device)
    optimizer = get_optimizer(model, args.learning_rate)
    num_training_steps = int(
        len(train_loader) * args.epochs // args.gradient_accumulation_steps
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_training_steps),
        num_training_steps=num_training_steps,
    )
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save

    # 如果由于意外,训练过程中断了，还可以重新恢复接着训练
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            filename = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            # Sorts folders by date modified, most recent checkpoint is the last
            dirs.sort(key=os.path.getctime)
            filename = dirs[-1]

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(filename)[0]

        if "epoch" in training_difference:
            # 比如上次训练完第三个epoch中断了，此时就可以直接在原来的基础上从第4个epoch训练开始
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_loader)
            resume_step -= starting_epoch * len(train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We need to skip steps until we reach the resumed step
            train_loader = accelerator.skip_first_batches(train_loader, resume_step)
            overall_step += resume_step

        for step, batch_dict in enumerate(tqdm(train_loader, desc="进行训练"), start=1):
            outputs = model(**batch_dict)
            train_loss += outputs.loss.detach().float()
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            overall_step += 1
            if isinstance(checkpointing_steps, int):
                output_dir = f"step_{overall_step}"
                if overall_step % checkpointing_steps == 0:
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
        # print("GPU上分配的样本数量", count, accelerator.process_index)
        train_loss /= len(train_loader.sampler)
        model.eval()
        valid_loss = 0
        preds = []
        trues = []
        for batch_dict in tqdm(valid_loader, desc="进行评估"):
            with torch.no_grad():
                outputs = model(**batch_dict)
            valid_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch_dict["labels"])
            )
            preds.append(predictions)
            trues.append(references)
        eval_metric = get_metrics(preds, trues)
        # 只在主进程上打印输出
        accelerator.print(f"epoch {epoch}:", eval_metric)
        accelerator.print(
            f"epoch:{epoch} ||train_loss:{train_loss} ||valid_loss:{valid_loss}"
        )
        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric["accuracy"],
                    "f1": eval_metric["f1"],
                    "train_loss": train_loss,
                    "epoch": epoch,
                },
                step=epoch,
            )
        # 需要选取指标最高的checkpoint
        if checkpointing_steps == "epoch":
            if args.output_dir is not None:
                output_dir = f"epoch_{epoch}"
                output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        if args.with_tracking:
            accelerator.end_training()

    accelerator.wait_for_everyone()


def get_args():
    base_args = Namespace(
        pretrained_model_path="/home/lis/algProjects/pretrained_models/chinese_wwm_ext_pytorch/",
        train_data_filepath="../Data/train.json",
        valid_data_filepath="../Data/valid.json",
        test_data_filepath="../Data/test.json",
        train_batch_size=32,
        eval_batch_size=64,
        learning_rate=3e-5,
        max_seq_len=256,
        warmup_proportion=0.1,
        gradient_accumulation_steps=2,
        epochs=3,
        seed=678,
        mixed_precision="fp16",
        use_cpu=False,
    )
    parser = ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--use_cpu", action="store_true", help="If passed, will train on the CPU."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    """
        The checkpoint save strategy to adopt during training. Possible values are:
    
            - `"no"`: No save is done during training.
            - `"epoch"`: Save is done at the end of each epoch.
            - `"steps"`: Save is done every `save_steps`.
    """

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        default=True,
        type=bool,
        # action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="checkpoint文件夹存储目录，默认为当前工作目录",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    args = parser.parse_args()
    for key, value in base_args.__dict__.items():
        setattr(args, key, value)
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

"""
代码来源：https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py
教程：https://huggingface.co/docs/accelerate/basic_tutorials/launchpip install deepspeed
"""


# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./acce_no_ds.yaml acce_no_ds_run.py
