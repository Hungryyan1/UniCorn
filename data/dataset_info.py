# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .interleave_datasets import MultiThinkInterleaveT2IDataset

DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    't2i_sft': T2IIterableDataset,
    't2i_reflect': MultiThinkInterleaveT2IDataset
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'data/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'data/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'data/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
        'self_edit': {
            'data_dir': 'data_dir',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'json_path',
        }
    },
    't2i_sft':{
            'prompt_with_persona_bagel_reward_50k_sample_5k': {
                'data_dir': 'data_dir', 
            'num_files': 1, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 10000, # number of total samples in the dataset
        }
    },
    'vlm_sft': {
        'map_judge': {
			'data_dir': 'image_dir',
			'jsonl_path': 'caption.jsonl',
			'num_total_samples': 3000
		},
        'self_caption': {
            'data_dir': 'image_dir',
            'jsonl_path': 'caption.jsonl',
            'num_total_samples': 5000
        }
    },
}