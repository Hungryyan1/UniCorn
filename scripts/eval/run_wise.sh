# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# generate images
torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    ./eval/gen/gen_images_mp_wise.py \
    --output_dir $output_path/images \
    --metadata_file ./eval/gen/wise/final_data.json \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \
    --think


# calculate score
python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/cultural_common_sense.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/spatio-temporal_reasoning.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/gpt_eval_mp.py \
        --json_path eval/gen/wise/data/natural_science.json \
        --image_dir $output_path/images \
        --output_dir $output_path

python3 eval/gen/wise/cal_score.py \
        --output_dir $output_path