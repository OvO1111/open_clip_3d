# 64k batchsize for 2.048e-3 lr
TORCH_CUDNN_V8_API_ENABLED=1 torchrun --nproc_per_node 1 ./training/main.py \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data datasets.ruijin.CLIPDataset \
    --dataset-type 3d \
    --lr "2.048e-3" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 782 \
    --wd 0.2 \
    --batch-size 64 \
    --epochs=100 \
    --workers=64 \
    --model RN101_3D \
    --precision 'amp_bf16' \
    --local-loss \
    --gather-with-grad \
    --force-image-size 64 128 128 \
    --grad-checkpointing \
    --log-every-n-steps 32 \
    --seed 0 \
    --logs ./logs/ \
    --name 'test' \
    --report-to "wandb" \
    --wandb-project-name "test"


