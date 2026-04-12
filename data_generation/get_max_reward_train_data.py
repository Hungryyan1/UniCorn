import json
import random
import base64
from pathlib import Path
import os.path as osp
import pandas as pd

image_root = 'path_to_images'

def extract_score(response):
    try:
        response = response.strip("```json\n")
        response = response.strip("\n```")
        response_data = json.loads(response)
        return response_data.get('score')
    except Exception as e:
        return None

def extract_prompt_text(prompt):
    # marker = "The prompt for image generation:"
    marker = "**Prompt:**\n"
    if marker in prompt:
        text = prompt.split(marker)[1].split("\n- **Generated Image")[0].strip()
        return text
    return None

def read_image_as_bytes(image_path):
    with open(image_path, 'rb') as f:
        return f.read()

def process_jsonl(input_file, output_file, log_file='processing_log.txt'):
    skipped_low_score = {}  # {prompt: count}
    skipped_parse_error = []  # [(idx, reason)]
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_data = []
    for line in lines:
        all_data.append(json.loads(line))
    
    results = []
    total_groups = len(all_data) // 8
    
    for group_idx in range(total_groups):
        start_idx = group_idx * 8
        end_idx = start_idx + 8
        group = all_data[start_idx:end_idx]
        
        valid_items = []
        for item in group:
            score = extract_score(item.get('response', ''))
            if score is not None:
                valid_items.append((item, int(score)))
            else:
                skipped_parse_error.append(
                    (item.get('idx'), 'Cannot extract score from response')
                )
        
        if not valid_items:
            prompt_text = extract_prompt_text(group[0].get('prompt', ''))
            print(f"Group {group_idx}: No valid items, skipping")
            continue
        
        max_score = max(item[1] for item in valid_items)
        
        if max_score <= 6:

            prompt_text = group[0].get('gen_prompt', '')
            if prompt_text:
                skipped_low_score[prompt_text] = skipped_low_score.get(prompt_text, 0) + 1
            print(f"Group {group_idx}: Max score {max_score} <= 6, skipping")
            continue
        
        max_items = [item for item in valid_items if item[1] == max_score]
        
        selected_item, selected_score = random.choice(max_items)
        
        prompt_text = selected_item.get('gen_prompt', '')
        image_path = selected_item.get('image', '')
        image_path = osp.join(image_root, image_path)
        major_category = selected_item.get('major_category', '')
        sub_category = selected_item.get('subcategory', '')

        
        try:
            image_bytes = read_image_as_bytes(image_path)
        except Exception as e:
            skipped_parse_error.append(
                (selected_item.get('idx'), f'Cannot read image: {e}')
            )
            print(f"Group {group_idx}: Cannot read image {image_path}, skipping")
            continue

        result = {
            "image": image_bytes,
            "captions": json.dumps({"caption": prompt_text}),
            'major_category': major_category,
            'sub_category': sub_category
        }
        
        results.append(result)
        print(f"Group {group_idx}: Selected idx {selected_item.get('idx')} with score {selected_score}")
    df = pd.DataFrame(results)
    df.to_parquet(output_file, engine='pyarrow', index=False)
    

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Processing Summary ===\n")
        f.write(f"Total groups processed: {total_groups}\n")
        f.write(f"Successfully saved: {len(results)}\n\n")
        
        f.write(f"=== Skipped due to low score (<=5): {len(skipped_low_score)} groups ===\n")
        for prompt, count in skipped_low_score.items():
            f.write(f"Count: {count}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write("-" * 80 + "\n")
        
        f.write(f"\n=== Skipped due to parse errors: {len(skipped_parse_error)} items ===\n")
        for idx, reason in skipped_parse_error:
            f.write(f"idx {idx}: {reason}\n")

    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Total groups: {total_groups}")
    print(f"Successfully saved: {len(results)}")
    print(f"Skipped (low score): {len(skipped_low_score)}")
    print(f"Skipped (parse errors): {len(skipped_parse_error)}")
    print(f"Details saved to: {log_file}")
    print(f"{'='*60}")
    return df


if __name__ == "__main__":
    input_file = 'input_jsonl'
    output_file = 'output_parquet'
    log_file = 'log_file.txt'

    df = process_jsonl(input_file, output_file, log_file=log_file)

    print("\nData preview:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Image dtype: {df['image'].dtype}")
    print(f"First image type: {type(df['image'].iloc[0])}")