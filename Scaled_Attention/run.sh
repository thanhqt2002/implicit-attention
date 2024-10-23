CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet/ --output_dir /path/to/output/directory/
