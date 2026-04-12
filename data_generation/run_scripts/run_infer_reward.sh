torchrun --nproc_per_node=8 \
    data_generation/reward/reward_infer.py \
    --input-file $input_path \
    --output-file $output_path \
    --image-root $image_root \
    --model-path $model_path \
    --max-new-tokens 2048
