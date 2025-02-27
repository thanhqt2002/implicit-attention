#!/bin/bash

count=0
offset=1800

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /home/ubuntu/shared/imagenet --output_dir ../outputs/imnet_attack/ --project_name 'implicit-attention-attack' --job_name implicit-pgd --attack 'pgd' --eps $eps --finetune /home/ubuntu/21hoang.p/khoa/implicit-attention/outputs/imnet/16_02_2025_14:58:47_deit_tiny_patch16_224_checkpoint.pth --eval 1 --robust --num_iter 2 --layer -1 --lambd 4 --use_wandb 1 --API_Key 37aa81100c443c3b5e4053d7732317a3953fe80d
done