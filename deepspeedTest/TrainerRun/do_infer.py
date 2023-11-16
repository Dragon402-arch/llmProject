import json
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from trainer_run import (
    compute_metrics,
    get_args,
    HotelReviewDataCollator,
    HotelReviewDataset,
)

model = BertForSequenceClassification.from_pretrained("./trained_models/checkpoint-114")

args = get_args()
data_collator = HotelReviewDataCollator(
    max_seq_len=args.max_seq_len, pretrained_model_path=args.pretrained_model_path
)

training_args = TrainingArguments(
    output_dir="trained_models",
    do_predict=True,
    remove_unused_columns=False,  # 如果想使用自定义的data_collator,就需要保证该参数为False，其默认为True
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

test_dataset = HotelReviewDataset(args.valid_data_filepath)
outputs = trainer.predict(test_dataset)
# print(outputs.predictions, outputs.label_ids)
print(json.dumps(outputs.metrics, ensure_ascii=False, indent=2))
