#  -*- coding: utf-8 -*-


def download_qwen_14b():
    from modelscope.hub.snapshot_download import snapshot_download

    model_dir = snapshot_download(
        "qwen/Qwen-14B-Chat",
        cache_dir="/date/pretrained_models/Qwen-14B",
        revision="v1.0.4",
    )


def test_qwen_14b():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig

    pretrained_model_path = "/date/pretrained_models/Qwen-14B-Chat/"

    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path, trust_remote_code=True
    )

    # use fp16
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path, device_map="auto", trust_remote_code=True, fp16=True
    ).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(
        pretrained_model_path, trust_remote_code=True
    )  # 可指定不同的生成长度、top_p等相关超参

    # 第一轮对话 1st dialogue turn
    response, history = model.chat(tokenizer, "你好", history=None)
    print(response)
    response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
    print(response)
