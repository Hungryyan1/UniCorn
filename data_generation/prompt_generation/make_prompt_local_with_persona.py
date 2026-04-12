# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import itertools
import json
import os
import csv
import random
import copy
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import datetime
import glob
import torch
from PIL import Image
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from tqdm import tqdm


LOGICAL_BATCH_SIZE = 1
# set a task inflation factor to generate more tasks than needed, to cover potential parsing failures
TASK_INFLATION_FACTOR = 2

CATEGORY_TARGET_COUNTS = {
    "Relational Operations": 20000,
    "Natural Science": 5000,
    "General Knowledge": 5000,
    "Spatio Reasoning": 9000,
    "Temporal Reasoning": 5000,
    "General Object": 24000,
    "Portrait": 9000,
    "Text Rendering": 11000,
    "Stylization": 9000,
    "Counting": 3000,
}      

CATEGORY_RULES = {
    "Relational Operations": {
      "name": "Relational Operations",
      "rule": "The prompt must include at least one logical relation between objects or people (such as action, comparison, contrast, or negation), and this relation should be the main task."
    },
    "Natural Science": {
      "name": "Natural Science",
      "rule": "The prompt must center on a real-world natural science phenomenon or entity (such as plants, animals, physics, or chemistry), with understanding or identifying this scientific element as the main task."
    },
    "General Knowledge": {
      "name": "General Knowledge",
      "rule": "The prompt must highlight a recognizable cultural element (such as a festival, sport, religious symbol, craft, or iconic figure/object), with recognizing or understanding this cultural element as the main task."
    },
    "Spatio Reasoning": {
      "name": "Spatio Reasoning",
      "rule": "The prompt must emphasize visible spatial relationships (such as left/right, front/behind, above/below, near/far, or occlusion), with reasoning or judging these spatial relations as the main task."
    },
    "Temporal Reasoning": {
      "name": "Temporal Reasoning",
      "rule": "The prompt must involve time-related information (such as simultaneous events, season, time of day, or historical period), with inferring or identifying this temporal information from a single image as the main task."
    },
    "General Object": {
      "name": "General Object",
      "rule": "The prompt must focus on non-human objects and their visible properties (such as shape, color, material, or arrangement), with observing, comparing, or describing these objects as the main task."
    },
    "Portrait": {
      "name": "Portrait",
      "rule": "The prompt must focus on human subjects, highlighting face, pose, clothing, or emotion, with recognizing or understanding the person’s characteristics as the main task."
    },
    "Text Rendering": {
      "name": "Text Rendering",
      "rule": "The prompt must require clearly legible text in the image (no more than two words) and must explicitly specify the exact text to display, making text rendering part of the main task."
    },
    "Stylization": {
      "name": "Stylization",
      "rule": "The prompt must specify a clear visual art style and take presenting characters or objects in this style (including appearance, colors, and background) as the main task."
    },
    "Counting": {
      "name": "Counting",
      "rule": "The prompt must require explicit counting of visible objects, with any counted category having a quantity of three or fewer, and the counting result being the main task."
    }
}


def load_seed_prompts(csv_path: str) -> Dict[str, List[Dict[str, str]]]:
    subcategory_data = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)        
        for row in reader:
            major_category = row.get('Major Category', '').strip()
            subcategory = row.get('Subcategory', '').strip()
            seed = row.get('Seed', '').strip()
            
            if not major_category or not subcategory or not seed:
                continue
            if subcategory not in subcategory_data:
                subcategory_data[subcategory] = []
            subcategory_data[subcategory].append({
                "major_category": major_category,
                "subcategory": subcategory,
                "seed": seed
            })
    return subcategory_data

def load_personas(jsonl_path: str) -> List[str]:
    personas = []
    if not jsonl_path or not os.path.exists(jsonl_path):
        print(f"Warning: Persona file not found at {jsonl_path}")
        return []
    
    print(f"Loading personas from {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if 'persona' in item and item['persona']:
                    personas.append(item['persona'])
            except Exception:
                continue
    print(f"Loaded {len(personas)} personas.")
    return personas

def count_existing_progress(output_file_path):

    counts = {cat: 0 for cat in CATEGORY_RULES.keys()}
    
    files_to_scan = [output_file_path]
    base_dir = os.path.dirname(output_file_path)
    base_name = os.path.basename(output_file_path)
    
    pattern = os.path.join(base_dir, f"{base_name}.rank_*")
    files_to_scan.extend(glob.glob(pattern))
    
    unique_prompts = set()

    for file_path in files_to_scan:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())

                        prompt_content = item.get('prompt', '').strip()
                        cat = item.get('major_category', '')
                        
                        if prompt_content and cat in counts:

                            if prompt_content not in unique_prompts:
                                counts[cat] += 1
                                unique_prompts.add(prompt_content)
                    except:
                        pass
    return counts


def get_subcategories_by_major_category(subcategory_data: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[str]]:
    major_to_subcategories = {}
    for subcat, items in subcategory_data.items():
        if items:
            major_cat = items[0]['major_category']
            if major_cat not in major_to_subcategories:
                major_to_subcategories[major_cat] = []
            if subcat not in major_to_subcategories[major_cat]:
                major_to_subcategories[major_cat].append(subcat)
    return major_to_subcategories

def sample_seeds_for_few_shot(subcategory_data, major_to_subcategories, target_major_category) -> Tuple[List[Dict], str]:
    valid_subcategories = major_to_subcategories.get(target_major_category, [])
    if not valid_subcategories:
        return [], "General"
    
    selected_subcategory = random.choice(valid_subcategories)
    seeds = subcategory_data.get(selected_subcategory, [])
    
    if not seeds:
         return [], selected_subcategory

    if selected_subcategory == "Anime":
        k = random.choice([1, 2])
    else:
        k = random.choice([1, 2, 3])
    k = min(k, len(seeds))
    
    selected_seeds = random.sample(seeds, k)
    
    few_shot_examples = []
    for seed_item in selected_seeds:
        few_shot_examples.append({
            "major_category": target_major_category,
            "subcategory": selected_subcategory,
            "prompt": seed_item['seed']
        })
    return few_shot_examples, selected_subcategory

def build_dynamic_system_prompt(current_examples_list: List[Dict], target_category_info: Dict, persona: str):
    category_rule_str = f"- **{target_category_info['name']}**: {target_category_info['rule']}\n"
    examples_json = json.dumps(current_examples_list, indent=2, ensure_ascii=False)

    return f"""
You are an expert dataset generator for prompt Bench. Your goal is to generate unique, challenging text-to-image prompts.

Your persona: {persona}

**Constraints:**
1. **Format**: Strictly return a valid JSON list.
2. **Task**: Generate English prompts ONLY for the specific category requested by the user. The prompt MUST adhere to the rules specified below for that category.
3. **Fields**: Each object must have 'major_category', 'subcategory', and 'prompt' fields.
4. **Information Priority**: The 'prompt' **must contain sufficient descriptive detail** to ensure complex image generation. **Do not prioritize brevity over informational density.**
5. **Persona**: The question must strictly exclude any names, occupations, professional skills, or backgrounds from the persona.

**Category Definition and Specific Rule (MUST FOLLOW THE RULE FOR THE TARGET CATEGORY):**
{category_rule_str}

**Current Reference Examples (Study the format and style):**
{examples_json}
"""

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    try:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return []

def collate_fn(batches):
    metadata = [_['metadata'] for _ in batches]
    return metadata

class PromptGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, seed_csv_path, output_file_path, persona_file_path):
        self.tasks = []
        self.subcategory_data = load_seed_prompts(seed_csv_path)
        self.major_to_subcategories = get_subcategories_by_major_category(self.subcategory_data)
        self.personas = load_personas(persona_file_path) if persona_file_path else []

        self.current_counts = count_existing_progress(output_file_path)

        TASK_INFLATION_FACTOR = 2.0 
        
        print(f"\n=== Dataset Initialization (Rank {torch.distributed.get_rank()}) ===")

        if torch.distributed.get_rank() == 0:
            print("Current progress (Merged + Partials):")
        
        for cat in CATEGORY_TARGET_COUNTS.keys():
            target = CATEGORY_TARGET_COUNTS.get(cat, 0)
            count = self.current_counts.get(cat, 0)
            
            if torch.distributed.get_rank() == 0:
                print(f"  {cat}: {count}/{target}")
            
            if count < target:
                needed = target - count
                estimated_calls = (needed * TASK_INFLATION_FACTOR + LOGICAL_BATCH_SIZE - 1) // LOGICAL_BATCH_SIZE
                estimated_calls = int(estimated_calls)
                if estimated_calls == 0 and needed > 0: estimated_calls = 1
                
                for _ in range(estimated_calls):
                    self.tasks.append(cat)
        
        random.seed(42 + len(self.tasks))
        random.shuffle(self.tasks)
        
        if torch.distributed.get_rank() == 0:
            print(f"Total tasks queued for this loop: {len(self.tasks)}")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        target_category_key = self.tasks[idx]
        target_category_info = CATEGORY_RULES[target_category_key]
        few_shot_examples, selected_subcategory = sample_seeds_for_few_shot(
            self.subcategory_data, self.major_to_subcategories, target_category_key
        )
        current_persona_text = ""
        if self.personas:
            current_persona_text = random.choice(self.personas)
        system_prompt = build_dynamic_system_prompt(few_shot_examples, target_category_info, current_persona_text)
        user_content = (
            f"Generate exactly {LOGICAL_BATCH_SIZE} new prompts. "
            f"Target Major Category: **{target_category_key}**. "
            f"Target Subcategory: **{selected_subcategory}**. "
            f"Rule to follow: **{target_category_info['rule']}**. "
            f"Each generated item must have 'major_category' field set to '{target_category_key}', 'subcategory' field set to '{selected_subcategory}', and 'prompt' field. "
            "Ensure high diversity and strictly adhere to the rule."
        )
        full_prompt_text = f"{system_prompt}\n\n{user_content}"
        return {
            'metadata': {
                'prompt_text': full_prompt_text,
                'target_category': target_category_key,
                'target_subcategory': selected_subcategory
            }
        }

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):

        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def generate_data_with_local_model():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    rank_output_file = f"{args.output_file}.rank_{rank}"
    
    loop_count = 0
    max_loops = 20 
    
    while True:
        loop_count += 1
        if rank == 0:
            print(f"\n========== Generation Loop {loop_count} ==========")

        dataset = PromptGenerationDataset(
            seed_csv_path=args.seed_csv,
            output_file_path=args.output_file,
            persona_file_path=args.persona_file_path,
        )
        
        if len(dataset) == 0:
            print(f"Rank {rank}: Local dataset empty. Targets likely met.")
            break
        
        if loop_count > max_loops:
            print(f"Rank {rank}: Max loops reached.")
            break

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        print(f"Rank {rank}: Start processing {len(dataloader)} batches...")
        
        for _, metadata_list in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"R{rank} L{loop_count}", position=rank):
            meta = metadata_list[0]
            raw_prompt = meta['prompt_text']
            
            try:
                response = model.chat(
                    tokenizer, 
                    new_token_ids,
                    image_transform,
                    images=[],
                    prompt=raw_prompt,
                    max_length=args.max_new_tokens,
                    temperature=0.8,
                    do_sample=True 
                )
                generated_list = extract_json_from_text(response)
            except Exception as e:
                print(f'[Rank {rank}] Error: {e}')
                generated_list = []

            valid_items = []
            if isinstance(generated_list, list):
                valid_items = generated_list
            elif isinstance(generated_list, dict):
                 for k, v in generated_list.items():
                     if isinstance(v, list):
                         valid_items.extend(v)
            
            if valid_items:
                with open(rank_output_file, 'a', encoding='utf-8') as f:
                    for item in valid_items:
                        if isinstance(item, dict) and 'prompt' in item:
                            item['major_category'] = meta['target_category']
                            item['subcategory'] = meta['target_subcategory']
                            
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Rank {rank}: Loop {loop_count} finished. Waiting for others...")
        torch.distributed.barrier()

    print(f"Rank {rank}: All loops finished.")
    torch.distributed.barrier()

    if rank == 0:
        print("Rank 0: Merging all rank files into final output...")
        final_items = []

        if os.path.exists(args.output_file):
            with open(args.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        final_items.append(json.loads(line.strip()))
                    except: pass

        base_dir = os.path.dirname(args.output_file)
        base_name = os.path.basename(args.output_file)
        pattern = os.path.join(base_dir, f"{base_name}.rank_*")
        rank_files = glob.glob(pattern)
        
        for r_file in rank_files:
            with open(r_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        final_items.append(json.loads(line.strip()))
                    except: pass
        
        merged_counts = {cat: 0 for cat in CATEGORY_RULES.keys()}
        final_output_list = []
        
        current_id = 0
        
        for item in final_items:
            cat = item.get('major_category')
            if cat not in CATEGORY_TARGET_COUNTS:
                continue
            
            target = CATEGORY_TARGET_COUNTS[cat]
            if merged_counts[cat] < target:
                item['id'] = current_id
                final_output_list.append(item)
                merged_counts[cat] += 1
                current_id += 1
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in final_output_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Merged successfully. Final counts: {merged_counts}")
        
        for r_file in rank_files:
            os.remove(r_file)
        print("Cleaned up rank files.")
    
    torch.distributed.barrier()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision-Language Model Prompt Generation')
    parser.add_argument('--model-path', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for generation')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of worker threads for data loading')
    parser.add_argument('--max-new-tokens', type=int, default=2048, help='Maximum number of new tokens to generate for each prompt')
    parser.add_argument('--seed-csv', type=str, required=True, help='CSV file path containing seed prompts with major and subcategory information')
    parser.add_argument('--output-file', type=str, required=True, help='Output JSONL file path')
    ## Optional argument for persona file path. We used personas from https://huggingface.co/datasets/proj-persona/PersonaHub/viewer, to further enhance the diversity of generated prompts.
    parser.add_argument('--persona-file-path', type=str, required=True, help='Persona file path')

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=datetime.timedelta(hours=6)
    )

    local_rank = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    print(f'Rank {torch.distributed.get_rank()}: Loading model...')
    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    generate_data_with_local_model()
