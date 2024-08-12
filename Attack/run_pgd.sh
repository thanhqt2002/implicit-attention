#!/bin/bash

count=0
offset=1600

for i in {6..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 \
     --master_port=$port_num --use_env main.py --model deit_tiny_patch16_224 --batch-size 48 \
     --data-path /root/tensorflow_datasets/downloads/manual/ --output_dir /root/checkpoints/attack/ \
     --project_name 'neurips_kpca_attack' --job_name rpc4it_tiny_pgd --attack 'pgd' --eps $eps \
     --finetune /root/checkpoints/experiments/rpc_tiny/4itperlayer_tiny.pth --eval 1 --use_wandb 1 \
     --API_Key f9b91afe90c0f06aa89d2a428bd46dac42640bff \
     --robust --num_iter 4 --layer -1 --lambd 3.0
done

# --resume /root/checkpoints/experiments/rpc_base/05_08_2024_13:32:23_deit_base_patch16_224_checkpoint.pth \
# --robust --num_iter 2 --lambd 3.0 --layer 0
#  --attack 'pgd'

# /root/checkpoints/experiments/rpc_base/4it_base.pth \
# --robust --num_iter 1 --lambd 3.2 --layer 12