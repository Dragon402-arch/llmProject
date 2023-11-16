import time
import os
import gc
import json
import torch
from transformers import BertForSequenceClassification, BertConfig
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm
import argparse
import torch.distributed as dist
import deepspeed
import sys

sys.path.append('/home/lis/algProjects/hotelReviewCls/')
from build_dataset import HotelReviewDataset, HotelReviewDataCollator

gc.enable()


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    # local_rank 参数在多卡训练时会自动变为对应的rank值。
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--pretrained_model_path",
        default="/home/lis/algProjects/pretrained_models/chinese_wwm_ext_pytorch",
        type=str,
    )
    parser.add_argument(
        "--train_data_filepath",
        default="../Data/train.json",
        type=str,
    )
    parser.add_argument(
        "--valid_data_filepath",
        default="../Data/valid.json",
        type=str,
    )
    parser.add_argument("--checkpoint_dir", default="trained_models", type=str)
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)

    # Include DeepSpeed configuration arguments
    # 添加部分deepspeed相关参数，其中包含 deepspeed_config 参数，可以不定义，在命令行也能传入接收
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(args.__dict__, ensure_ascii=False, indent=2))
    return args


args = get_args()

deepspeed.init_distributed(dist_backend='nccl')
# args.local_rank = int(os.environ['LOCAL_RANK'])
# args.local_rank = model_engine.local_rank
# device = (torch.device("cuda", args.local_rank) if (args.local_rank > -1)
#               and torch.cuda.is_available() else torch.device("cpu"))


train_dataset = HotelReviewDataset(args.train_data_filepath)
eval_dataset = HotelReviewDataset(args.valid_data_filepath)
data_collator = HotelReviewDataCollator(
    max_seq_len=args.max_seq_len, pretrained_model_path=args.pretrained_model_path
)

# 如果训练的模型在分片后能放得下，但在分片前放不下，则需要使用该方法。
# with deepspeed.zero.Init():
#     config = BertConfig.from_pretrained(args.pretrained_model_path)
#     model = BertForSequenceClassification(config=config)
model = BertForSequenceClassification.from_pretrained(args.pretrained_model_path)


# model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

DEEPSPEED_CONFIG = {
    'fp16': {
        'enabled': True,
        "loss_scale": 128,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    'optimizer': {
        'type': 'AdamW',
        'params': {
            'lr': 3e-05,
            'betas': [0.9, 0.999],
            'eps': 1e-08,
            'weight_decay': 0.0,
        },
    },
    'scheduler': {
        'type': 'WarmupLR',
        'params': {'warmup_min_lr': 0, 'warmup_max_lr': 1e-05, 'warmup_num_steps': 100},
    },
    'zero_optimization': {
        'stage': 2,
        'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
        'offload_param': {'device': 'cpu', 'pin_memory': False},
    },
    'train_batch_size': 128,
    'train_micro_batch_size_per_gpu': 32,
    'gradient_accumulation_steps': 2,
}


# 没有传入配置文件时，使用dict中的配置，否则使用传入配置文件中的配置。
if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
    DEEPSPEED_CONFIG = None

model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
    args=args,
    config=DEEPSPEED_CONFIG,
    model=model,
    model_parameters=model.parameters(),  # model_parameters,
    training_data=train_dataset,
    collate_fn=data_collator,
)

# Overwrite application configs with DeepSpeed config
args.train_micro_batch_size_per_gpu = model_engine.train_micro_batch_size_per_gpu()
# args.gradient_accumulation_steps = model_engine.gradient_accumulation_steps()


eval_loader = model_engine.deepspeed_io(
    eval_dataset,
    batch_size=args.train_micro_batch_size_per_gpu,
    route="eval",
    collate_fn=data_collator,
)


def gather_trues_preds(trues: list, preds: list):
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    all_true_labels = [torch.ones_like(trues) for _ in range(dist.get_world_size())]
    all_pred_labels = [torch.ones_like(preds) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=all_pred_labels, tensor=preds)
    dist.all_gather(tensor_list=all_true_labels, tensor=trues)
    preds = torch.cat(all_pred_labels, dim=0)
    trues = torch.cat(all_true_labels, dim=0)
    return trues, preds


def main(args, model_engine, train_loader, eval_loader):
    for epoch in range(1, args.num_epochs + 1):
        train_loss = 0
        model_engine.train()
        for step, batch_dict in enumerate(tqdm(train_loader, desc="进行训练")):
            batch_dict = {
                key: value.to(model_engine.local_rank)
                for key, value in batch_dict.items()
            }
            loss = model_engine(**batch_dict).loss
            train_loss += loss
            model_engine.backward(loss)
            model_engine.step()

        model_engine.eval()
        valid_loss = 0
        preds, trues = [], []
        for batch_dict in tqdm(eval_loader, desc="进行评估", disable=dist.get_rank() != 0):
            batch_dict = {
                key: value.to(model_engine.local_rank)
                for key, value in batch_dict.items()
            }
            with torch.no_grad():
                model_outputs = model_engine(**batch_dict)
                loss = model_outputs.loss
                predictions = model_outputs.logits.argmax(dim=-1)
                preds.append(predictions)
                trues.append(batch_dict["labels"])
                valid_loss += loss
        trues, preds = gather_trues_preds(trues, preds)
        # 默认求和，将损失值求和结果保存在rank=0上面
        dist.reduce(valid_loss, dst=0)
        # 默认求和，将损失值求和结果保存在每个rank上面
        # dist.all_reduce(valid_loss)
        dist.reduce(train_loss, dst=0)
        model_engine.save_checkpoint(
            save_dir=args.checkpoint_dir, client_state={'checkpoint_epoch': epoch}
        )
        if dist.get_rank() == 0:
            y_true = trues.detach().cpu().numpy()
            y_pred = preds.detach().cpu().numpy()
            precision, recall, f_score, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="macro"
            )
            print(classification_report(y_true=y_true, y_pred=y_pred))
            print(
                f"epoch:{epoch} \t train_loss:{train_loss.item()} \t valid_loss:{valid_loss.item()} \
                    \t precision:{precision:.3f} \t recall:{recall:.3f} \t f_score:{f_score:.3f}"
            )
        dist.barrier()


if __name__ == "__main__":
    main(args, model_engine, train_loader, eval_loader)
    gc.collect()


# deepspeed --include localhost:0,1 deepspeed_run.py --deepspeed_config ds_config.json
