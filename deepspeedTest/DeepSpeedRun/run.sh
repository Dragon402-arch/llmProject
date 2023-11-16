deepspeed --include localhost:0,1 deepspeed_run.py --deepspeed_config ds_config.json

# 不传入deepspeed_config.json文件，则使用文件内dict配置
# deepspeed --include localhost:0,1 deepspeed_run.py



# https://github.com/microsoft/DeepSpeedExamples/blob/master/training/bing_bert/deepspeed_train.py

# 欠缺保存/加载模型，以及使用模型进行推理

# 模型每50step保存一次，方便训练中断之后恢复到最近一次接着训练。