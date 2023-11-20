import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer, AutoModel

pretrained_model_path = "/root/llmProjects/chatglm3"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(pretrained_model_path, device_map="auto", trust_remote_code=True).half()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
