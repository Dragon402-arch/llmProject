
# DataParallel DP

# export CUDA_VISIBLE_DEVICES=0,1,2

# python trainer_run.py


# DistributedDataParallel DDP
export CUDA_VISIBLE_DEVICES=0,1,2 

OMP_NUM_THREADS=10 torchrun --nproc_per_node=3 --master_port=6996 trainer_run.py 

# 该命令已经被弃用
# OMP_NUM_THREADS=10 python -m torch.distributed.launch --nproc_per_node=3 --master_port=6996 trainer_run.py

# DeepSpeed
# deepspeed --include localhost:0,1,2 trainer_run.py --deepspeed_config_file ds_config.json
