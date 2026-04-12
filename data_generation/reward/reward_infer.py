import argparse
import itertools
import json
import os
import shutil
from pathlib import Path
import datetime
import torch
from PIL import Image
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from tqdm import tqdm


def collate_fn(batches):
    images = [_['images'] for _ in batches]
    data_items = [_['data_item'] for _ in batches]
    return images, data_items


class JsonlDataset(torch.utils.data.Dataset):

    def __init__(self, jsonl_path, image_root=None):
        self.data = []
        self.image_root = image_root
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                item['idx'] = idx  
                image_path = item['image']
                if self.image_root:
                    image_path = os.path.join(self.image_root, image_path)
                
                if not os.path.exists(image_path):
                    continue 
                self.data.append(item)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'Loaded {len(self.data)} samples from {jsonl_path}')
        elif not torch.distributed.is_initialized():
            print(f'Loaded {len(self.data)} samples from {jsonl_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        
        image_path = data_item['image']
        if self.image_root:
            image_path = os.path.join(self.image_root, image_path)
        
        try:
            image = Image.open(image_path)
            image = image.convert('RGB') if image.mode != 'RGB' else image
        except Exception as e:
            print(f'Error loading image {image_path}: {e}')
            image = Image.new('RGB', (224, 224), color='white')
        
        images = [image]

        return {
            'images': images,
            'data_item': data_item,
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


def evaluate_chat_model():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    dataset = JsonlDataset(
        jsonl_path=args.input_file,
        image_root=args.image_root,
    )
    
    part_file = f"{args.output_file}.part{rank}"
    
    finished_indices = set()
    if os.path.exists(part_file):
        with open(part_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    res = json.loads(line.strip())
                    finished_indices.add(res['idx'])
                except Exception:
                    continue
        print(f"[Rank {rank}] Found {len(finished_indices)} processed samples in {part_file}")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    output_f = open(part_file, 'a', encoding='utf-8')

    for _, (images, data_items) in tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0)):
        data_item = data_items[0]
        idx = data_item['idx']

        if idx in finished_indices:
            continue

        prompt = data_item['prompt']
        
        images = images[0]
        images, conversation = process_conversation(images, prompt)

        try:
            response = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images,
                prompt=conversation,
                max_length=args.max_new_tokens,
            )
        except Exception as e:
            print(f'[Rank {rank}] Error during inference idx {idx}: {e}')
            response = f'ERROR: {str(e)}'

        output_item = {
            'idx': idx,
            'prompt': data_item['prompt'],
            'image': data_item['image'],
            'response': response,
        }
        
        for key in data_item:
            if key not in ['idx', 'prompt', 'image']:
                output_item[key] = data_item[key]
        
        output_f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        output_f.flush()

    output_f.close()
    
    print(f"[Rank {rank}] Finished.")
    torch.distributed.barrier()

    if rank == 0:
        print("Merging results from all ranks...")
        all_results = []
        
        for r in range(world_size):
            p_file = f"{args.output_file}.part{r}"
            if os.path.exists(p_file):
                with open(p_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            all_results.append(json.loads(line.strip()))
                        except:
                            continue

        all_results.sort(key=lambda x: x['idx'])

        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f'Final results saved to {args.output_file}')
        print(f'Total samples: {len(all_results)}')

        # #cleanup part files
        # for r in range(world_size):
        #     p_file = f"{args.output_file}.part{r}"
        #     if os.path.exists(p_file):
        #         os.remove(p_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision-Language Model Inference')
    parser.add_argument('--input-file', type=str, required=True,
                        help='input JSONL file path, each line should contain at least "prompt" and "image" fields')
    parser.add_argument('--output-file', type=str, required=True,
                        help='output JSONL file path')
    parser.add_argument('--image-root', type=str, default=None,
                        help='image root directory, if the image field is a relative path, this needs to be specified')
    parser.add_argument('--model-path', type=str, required=True,
                        help='model path')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='only support batch size 1')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of dataloader workers')
    parser.add_argument('--max-new-tokens', type=int, default=2048,
                        help='max new tokens to generate')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f'Input file not found: {args.input_file}')

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=datetime.timedelta(hours=3)
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if torch.distributed.get_rank() == 0:
        print('Loading model...')
    
    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if torch.distributed.get_rank() == 0:
        print(f'Total parameters: {total_params:.2f}B')

    evaluate_chat_model()
