torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    data_generation/prompt_generation/make_prompt_local_with_persona.py \
    --output-file $output_jsonl \
    --batch-size 1 \
    --seed-csv $seed_csv \
    --persona-file-path $persona_file_path \
    --model-path $model_path