###### FOR TRAINING
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 \
--use_env main_train.py \
--model deit_base_patch16_224 --batch-size 256 --data-path /path/to/data/imagenet/ \
--output_dir /path/to/checkpoints/ \
--clip-grad 1.0 \
--robust --num_iter 1 --lambd 4.0 --layer 0 

###### FOR ROBUSTNESS EVAL
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env eval_OOD.py \
--model deit_tiny_patch16_224 --data-path /path/to/data/imagenet/ --output_dir /path/to/checkpoints/ \
--robust --num_iter 4 --lambd 4 --layer 0 --resume /path/to/model/checkpoint/