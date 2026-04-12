set -x

torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    data_generation/image_generation/gen_images_mp.py \
    --output_dir $output_path/images \
    --metadata_file $prompt_path \
    --batch_size 1 \
    --num_images 8 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path