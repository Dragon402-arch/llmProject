#  -*- coding: utf-8 -*-
import math
import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import evaluate
from accelerate import Accelerator
from accelerate.utils import DummyScheduler, set_seed
from accelerate.logging import get_logger
from transformers import BertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace

from acce_ds_utils import get_optimizer, load_training_checkpoint, checkpoint_model

import sys

sys.path.append('/home/lis/algProjects/hotelReviewCls/')
from build_dataset import HotelReviewDataset, HotelReviewDataCollator


logger = get_logger(__name__)


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
    # Initialize accelerator
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.use_cpu,
            mixed_precision=args.mixed_precision,
            project_dir=args.logging_dir,
            device_placement=True,
            split_batches=False,
        )
    else:
        accelerator = Accelerator(
            cpu=args.use_cpu,
            mixed_precision=args.mixed_precision,
            device_placement=True,  # 不用再对model以及tensor进行指定了。
        )

    logger.info(accelerator.state, main_process_only=False)

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, args.__dict__)

    with accelerator.main_process_first():
        train_dataset = HotelReviewDataset(args.train_data_filepath)
        eval_dataset = HotelReviewDataset(args.valid_data_filepath)
    # test_dataset = HotelReviewDataset(args.test_data_filepath)

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
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    valid_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    set_seed(args.seed)

    model = BertForSequenceClassification.from_pretrained(args.pretrained_model_path)
    # model = model.to(accelerator.device)

    optimizer = get_optimizer(model, accelerator, args)

    accelerator.print(accelerator.state.deepspeed_plugin.deepspeed_config)

    #  # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        args.gradient_accumulation_steps = (
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ]
        )
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    num_warmup_steps = args.warmup_proportion * args.max_train_steps

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            total_num_steps=args.max_train_steps,
            warmup_num_steps=num_warmup_steps,
        )
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_metric = 0
    best_metric_checkpoint = None

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        _, last_global_step = load_training_checkpoint(
            model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        resume_step = last_global_step
        starting_epoch = resume_step // len(train_loader)
        resume_step -= starting_epoch * len(train_loader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(
            tqdm(
                train_loader, desc="进行训练", disable=not accelerator.is_local_main_process
            )
        ):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            # with accelerator.autocast():
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            train_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(
                train_loader
            ) - 1:
                optimizer.step()
                # 如果在混合精度训练时，由于初始loss_scale值较大使得梯度出现上溢时，不进行梯度更新，此时也要不进行学习率的变化。
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            # 方便中断之后接着训练
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        train_loss /= len(train_dataset)

        # print("GPU上分配的样本数量", count, accelerator.process_index)
        model.eval()
        valid_loss = 0
        preds = []
        trues = []
        for batch_dict in tqdm(
            valid_loader, desc="进行评估", disable=not accelerator.is_local_main_process
        ):
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
        valid_loss /= len(eval_dataset)
        # 只在主进程上打印输出
        accelerator.print(f"epoch {epoch},step {completed_steps}:", eval_metric)
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
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            # Save the DeepSpeed checkpoint to the specified path
            # 保存 32 bit model
            checkpoint_model(
                accelerator, output_dir, epoch, model, epoch, completed_steps
            )

        # New Code #
        # 需要选取指标最高的checkpoint
        if eval_metric["f1"] > best_metric:
            best_metric = eval_metric["f1"]
            best_metric_checkpoint = os.path.join(args.output_dir, str(epoch))
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")

    # Loads the best checkpoint after the training is finished
    if args.load_best_model:
        _, last_global_step = load_training_checkpoint(
            model,
            "/".join(best_metric_checkpoint.split("/")[:-1]),
            tag=best_metric_checkpoint.split("/")[-1],
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        # Evaluates using the best checkpoint
        model.eval()
        eval_loss = 0
        preds = []
        trues = []
        for batch_dict in tqdm(valid_loader, desc="进行评估"):
            with torch.no_grad():
                outputs = model(**batch_dict)
            eval_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch_dict["labels"])
            )
            preds.append(predictions)
            trues.append(references)
        eval_metric = get_metrics(preds, trues)
        logger.info(
            f"Best model metrics: eval_f1: {eval_metric['f1']} eval_loss: {eval_loss}"
        )
        if eval_metric['f1'] != best_metric:
            raise AssertionError(
                f"Best metric {best_metric} does not match the metric {eval_metric['f1']} of the loaded best model."
            )
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        #  accelerator.prepare()方法对model进行了包装，在保存模型权重时，需要去除其包装。
        # 保存 16bit model
        unwrapped_model = accelerator.unwrap_model(model)

        # New Code #
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

        # accelerator.save(unwrapped_model.state_dict(), args.output_dir)


def get_args():
    args = Namespace(
        pretrained_model_path="/home/lis/algProjects/pretrained_models/chinese_wwm_ext_pytorch/",
        train_data_filepath="../Data/train.json",
        valid_data_filepath="../Data/valid.json",
        test_data_filepath="../Data/test.json",
        learning_rate=3e-5,
        max_seq_len=256,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        warmup_proportion=0.1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        seed=678,
        mixed_precision="no",  # ["no", "fp16", "bf16", "fp8"]
        use_cpu=False,  # 使用CPU进行训练
        checkpointing_steps="epoch",  # ["no", "steps", "epoch"]
        with_tracking=True,
        output_dir="zero_stage3",  # 模型保存路径
        logging_dir="logs",
        resume_from_checkpoint=None,
        load_best_model=False,
        weight_decay=0.01,
        max_train_steps=None,
        lr_scheduler_type="linear"
        # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

"""
GPU版本:
    accelerate launch --config_file ./zero_stage3_acce_config.yaml acce_ds_run.py
CPU版本
    bug信息
        torch/include/ATen/cuda/CUDAContext.h:10:10: fatal error: cusolverDn.h: No such file or directory
    解决办法
        export CUDA_HOME=/usr/local/cuda-11.6
    export CUDA_HOME=/usr/local/cuda-11.6
    accelerate launch --config_file ./zero_stage3_acce_config.yaml acce_ds_run.py
"""

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./acce_ds_config.yaml acce_ds_config_run.py
