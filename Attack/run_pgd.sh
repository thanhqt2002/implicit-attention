#!/bin/bash

count=0
offset=1800

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /path/to/data/imagenet --output_dir /path/to/output/directory/ --project_name 'project_name' --job_name job_name --attack 'pgd' --eps $eps --finetune /path/to/trained/model/ --eval 1 --robust --num_iter 2 --layer -1 --lambd 4
done