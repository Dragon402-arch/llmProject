#  -*- coding: utf-8 -*-


def download_chatglm2():
    from modelscope.hub.snapshot_download import snapshot_download

    model_dir = snapshot_download(
        'ZhipuAI/chatglm2-6b',
        cache_dir='/home/lis/algProjects/pretrained_models/chatglm2-6b',
        revision="v1.0.2")


def test_chatglm2():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    from transformers import AutoTokenizer, AutoModel
    pretrained_model_path = "/home/lis/algProjects/pretrained_models/chatglm2-6b/"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(pretrained_model_path, device_map="auto", trust_remote_code=True).half()
    model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)
