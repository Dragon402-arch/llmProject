import os
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import torch
import deepspeed

pretrained_config_path = (
    "/home/lis/algProjects/pretrained_models/chinese_wwm_ext_pytorch"
)
trained_model_path = "trained_models/global_step85/mp_rank_00_model_states.pt"

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))


def get_model(pretrained_config_path, trained_model_path):
    config = BertConfig.from_pretrained(pretrained_config_path)
    model = BertForSequenceClassification(config)
    latest_checkpoint_path = trained_model_path
    state_dict = torch.load(latest_checkpoint_path)
    # print(state_dict.keys())
    model.load_state_dict(state_dict["module"], strict=True)
    return model


model = get_model(pretrained_config_path, trained_model_path)

model_engine = deepspeed.init_inference(
    model,
    mp_size=world_size,
    dtype=torch.float,
)
tokenizer = BertTokenizer.from_pretrained(pretrained_config_path)

review = "这家酒店真差劲，以后再也不来住了"
batch_dict = tokenizer(review, return_tensors="pt")
# print(batch_dict)
with torch.no_grad():
    batch_dict = {key: value.to(local_rank) for key, value in batch_dict.items()}
    logits = model_engine(**batch_dict).logits
    preds = logits.argmax(dim=-1)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(preds)


# deepspeed --include localhost:0,1 do_infer.py


# https://www.deepspeed.ai/tutorials/inference-tutorial/
