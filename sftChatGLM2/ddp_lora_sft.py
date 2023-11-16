#  -*- coding: utf-8 -*-

"""
自行编写模型训练过程，使用多GPU并行训练。
参考代码：https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b
"""
import warnings

warnings.filterwarnings("ignore")

import os
import random
import warnings
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

from build_dataset import ChatGLMDataset, ChatGLMDataCollator
from chatglm_model import ChatGLMForConditionalGeneration

# 指定数据路径与预训练模型路径
tasks = ["changeName", "IPHelper"]
task = "IPHelper"
assert task in tasks
if task == "changeName":
    train_data_filepath = "/home/lis/algProjects/finetuneChatGLM/train_data.json"
    valid_data_filepath = "/home/lis/algProjects/finetuneChatGLM/valid_data.json"
    pretrained_model_path = "/date/pretrained_models/chatglm2-6b/"
    checkpoint_filepath = "./chatglm2_name_lora"
else:
    train_data_filepath = "/home/lis/algProjects/finetuneChatGLM/trainChatGLM2/ip_data/ip_train_data.json"
    valid_data_filepath = "/home/lis/algProjects/finetuneChatGLM/trainChatGLM2/ip_data/ip_valid_data.json"
    pretrained_model_path = "/date/pretrained_models/chatglm2-6b/"
    checkpoint_filepath = "./chatglm2_ip_lora_final"

base_args = Namespace(
    train_data_filepath=train_data_filepath,
    valid_data_filepath=valid_data_filepath,
    pretrained_model_path=pretrained_model_path,
    checkpoint_filepath=checkpoint_filepath,
    # max_seq_len=256,
    max_seq_len=2048,
    train_batch_size=16,
    eval_batch_size=16,
    # batch_size=4,
    batch_size=8,
    learning_rate=3e-4,
    # learning_rate=5e-4,
    warmup_proportion=0.1,
    epochs=8,
    gradient_accumulation_steps=2,
    seed=666
)


def get_args():
    parser = ArgumentParser()
    # GPU配置参数
    parser.add_argument(
        "--gpu_devices", type=int, nargs="+", default=[0, 1, 2, 3], help=""
    )  # 改变量传递的是一个列表参数
    args = parser.parse_args()
    for key, value in base_args.__dict__.items():
        setattr(args, key, value)
    return args


def init_dist(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        "nccl", rank=rank, init_method="tcp://127.0.0.1:3457", world_size=world_size
    )


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


def get_model(pretrained_model_path, device):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['query_key_value', 'dense_h_to_4h', 'dense_4h_to_h'],
        fan_in_fan_out=False
    )
    model = ChatGLMForConditionalGeneration.from_pretrained(
        pretrained_model_path,
        # device_map="auto",
        revision="fp16"
    ).half()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    return model.to(device=device)


def main_worker(nprocs, args, ngpus_per_node):
    """
    Args:
        nprocs: nprocs 的取值是所使用的GPU索引数，可能的取值为0,1,2,3
    """
    init_dist(rank=nprocs, world_size=ngpus_per_node)

    # 将进程与指定的GPU进行对应，否则就会选取(0,1)/(0,1,2)的GPU组合，对应后可以选到(1,2)这种组合。
    gpu_idx = args.gpu_devices[nprocs]
    print("Use GPU: {} for training".format(gpu_idx))
    torch.cuda.set_device(gpu_idx)

    # 因为使用多个GPU时，批量大小实际上会增加，如果不调整学习率，可能会导致收敛速度变慢。
    args.learning_rate *= ngpus_per_node  # # 学习率要根据并行GPU的数量进行倍增
    args.batch_size = int(args.batch_size / ngpus_per_node)
    data_collator = ChatGLMDataCollator(args.max_seq_len, args.pretrained_model_path)
    train_dataset = ChatGLMDataset(args.train_data_filepath)
    valid_dataset = ChatGLMDataset(args.valid_data_filepath)

    # Default process group is initialized 只有在进程组初始化之后才能进行该处理，将数据集拆分为ngpus_per_node份，且无重复。
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=ngpus_per_node, rank=nprocs
    )
    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=0,  # 单卡训练时不选0，可以加快数据加载的速度。
        sampler=train_sampler,
        pin_memory=False,
        collate_fn=data_collator,
        drop_last=False,  # drop_last默认为False，也就是会使用已有的样本进行填充，保证每个GPU上分配的数据都是完整的batch_size
    )

    valid_sampler = DistributedSampler(
        valid_dataset, num_replicas=ngpus_per_node, rank=nprocs
    )
    valid_loader = DataLoader(
        valid_dataset,
        args.batch_size,
        shuffle=(valid_sampler is None),
        num_workers=0,
        sampler=valid_sampler,
        pin_memory=False,  # 看下这个参数是否有必要
        collate_fn=data_collator,
    )
    model = get_model(args.pretrained_model_path, device=gpu_idx)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu_idx], find_unused_parameters=False
    )
    optimizer = get_optimizer(model, learning_rate=args.learning_rate)
    train_and_eval(
        args, model, optimizer, gpu_idx, train_loader, valid_loader
    )
    dist.destroy_process_group()


def train_and_eval(args, model, optimizer, device, train_loader, valid_loader, loss_or_metric="loss"):
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
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        dist.barrier()
        for step, batch_dict in enumerate(tqdm(train_loader, desc="进行训练"), start=1):
            batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
            # autocast should wrap only the forward pass(es) of your network, including the loss computation(s).
            # Backward passes under autocast are not recommended.
            # Enables autocasting for the forward pass (model + loss)
            with torch.autocast("cuda"):
                loss = model(**batch_dict).loss
                loss = loss.mean()  # mean() to average on multi-gpu.
                train_loss += loss.item()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if step % 20 == 0:
                    print(loss.item())

            # backward  Exits the context manager before backward()
            # Accumulates scaled gradients.
            scaler.scale(loss).backward()
            # weights update
            if step % args.gradient_accumulation_steps == 0:
                # may unscale_ here if desired
                scaler.step(optimizer)
                # Internally invokes unscale_(optimizer) (unless unscale_() was explicitly called for optimizer earlier in the iteration).
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
            torch.cuda.empty_cache()
        train_loss = torch.as_tensor(train_loss, dtype=torch.float, device=device)
        dist.all_reduce(train_loss)
        train_loss = train_loss.item()
        train_loss /= (dist.get_world_size() * len(train_loader.sampler))
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_dict in tqdm(valid_loader, desc="进行评估"):
                batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
                with torch.cuda.amp.autocast():
                    loss = model(**batch_dict).loss
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    valid_loss += loss.item()
                    torch.cuda.empty_cache()
            valid_loss = torch.as_tensor(valid_loss, dtype=torch.float, device=device)
            dist.all_reduce(valid_loss)
            valid_loss = valid_loss.item()
            valid_loss /= (dist.get_world_size() * len(valid_loader.sampler))

            if dist.get_rank() == 0:
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
                                k: v.to("cpu") for k, v in model.module.named_parameters() if v.requires_grad
                            }
                            torch.save(saved_params, args.checkpoint_filepath)
                        else:
                            model.module.save_pretrained(args.checkpoint_filepath)
                elif loss_or_metric == "metric":
                    if eval_micro_f1 > best_eval_f1:
                        torch.save(model.module.state_dict(), args.checkpoint_filepath)
                        best_eval_f1 = eval_micro_f1
                else:
                    raise ValueError("在验证集上选取保存模型的方式有误。")
        model.train()


def exec_train(args):
    torch.cuda.empty_cache()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    #     [str(idx) for idx in args.gpu_devices]
    # )  # '0,1,2,3'

    # 放到开启多进程之前进行处理。
    # train_dataset = ChatGLMDataset(args.train_data_filepath)
    # valid_dataset = ChatGLMDataset(args.valid_data_filepath)
    # data_collator = ChatGLMDataCollator(args.max_seq_len, args.pretrained_model_path)

    ngpus_per_node = len(args.gpu_devices)  # torch.cuda.device_count()
    mp.spawn(
        main_worker,
        args=(args, ngpus_per_node),
        nprocs=ngpus_per_node,
    )  # nprocs:number process


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    exec_train(args)
