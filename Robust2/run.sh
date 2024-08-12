###### FOR TRAINING
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 2 --nproc_per_node=4 \
--use_env /root/repos/KPCA_code/Robust2/main_train.py \
--model deit_base_patch16_224 --batch-size 256 --data-path /root/tensorflow_datasets/downloads/manual/ \
--output_dir /root/checkpoints/experiments/rpc_base/ \
--clip-grad 1.0 \
--robust --num_iter 2 --lambd 2.0 --layer 0 \
--wandb --API_Key f9b91afe90c0f06aa89d2a428bd46dac42640bff \
--job_name base_rpc13_TEST6_resume2 \
--resume /root/checkpoints/experiments/rpc_base/05_08_2024_13:32:23_deit_base_patch16_224_checkpoint.pth --epochs 350

###### FOR ROBUSTNESS EVAL
# CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 2 --nproc_per_node=4 --use_env /root/repos/KPCA_code/Robust2/eval_OOD.py \
# --model deit_base_patch16_224 --data-path /root/datasets/ --output_dir /root/checkpoints/ \
# --resume /root/checkpoints/experiments/rpc_base/2it_base.pth \
# --robust --num_iter 2 --lambd 1.0 --layer 0

# /root/checkpoints/experiments/rpc_base/2it_base.pth \
# --robust --num_iter 2 --lambd 0.5 --layer 0
# /root/checkpoints/experiments/rpc_base/base_baseline2.pth

# --robust --num_iter 4 --lambd 4 --layer 0 