from argparse import ArgumentParser
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
import sys

sys.path.append('/home/lis/algProjects/hotelReviewCls/')
from build_dataset import HotelReviewDataset, HotelReviewDataCollator

import warnings

warnings.filterwarnings("ignore")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def freeze_weights(model):
    # Freeze the whole BERT model and train just the classifier
    for name, param in model.bert.named_parameters():
        param.requires_grad = False

    # Freeze BERT except the pooler layer
    for name, param in model.bert.named_parameters():
        if not name.startswith('pooler'):
            param.requires_grad = False

    # Freeze the first 23 layers of the BERT
    for name, param in model.bert.named_parameters():
        if (not name.startswith('pooler')) and "layer.23" not in name:
            param.requires_grad = False


def get_args():
    parser = ArgumentParser(description='Training')
    # local_rank 参数在多卡训练时会自动变为对应的rank值，不需要用户传入。
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
    parser.add_argument(
        "--test_data_filepath",
        default="../Data/valid.json",
        type=str,
    )
    parser.add_argument(
        "--deepspeed_config_file",
        default=None,
        type=str,
    )
    parser.add_argument("--checkpoint_dir", default="trained_models", type=str)
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)

    args = parser.parse_args()
    print(json.dumps(args.__dict__, ensure_ascii=False, indent=2))
    return args


# training_args = HfArgumentParser(
#      (TrainingArguments)
#  ).parse_args_into_dataclasses()


def main():
    args = get_args()
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.num_epochs,
        logging_steps=43,  # 该参数指定了模型多久会输出一次在验证集上的评价指标。
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="steps",
        remove_unused_columns=False,  # 该参数保证了可以使用自定义的data_collator
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed_config_file,
    )
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model_path)

    train_dataset = HotelReviewDataset(args.train_data_filepath)
    eval_dataset = HotelReviewDataset(args.valid_data_filepath)
    test_dataset = HotelReviewDataset(args.valid_data_filepath)
    data_collator = HotelReviewDataCollator(
        max_seq_len=args.max_seq_len, pretrained_model_path=args.pretrained_model_path
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        # compute_loss = None
    )
    trainer.train()


if __name__ == "__main__":
    # main()
    get_args()
