
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240