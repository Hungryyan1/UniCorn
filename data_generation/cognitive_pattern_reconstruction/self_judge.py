import json
import random
import os
import os.path as osp
import shutil

TOTAL_SAMPLES_NEEDED = 3000

image_source_root = 'image_source_root'
input_jsonl_file = 'input_jsonl_file'
###  input format example (from reward infer in the first stage):
###  {"id": 0, "image": "0.jpg", "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]}

output_dir = 'output_dir'
output_image_dir = osp.join(output_dir, 'images')
output_jsonl_file = osp.join(output_dir, 'captions.jsonl')


def get_start_id(file_path):
    if not osp.exists(file_path) or osp.getsize(file_path) == 0:
        return 0
    max_id = -1
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                max_id = max(max_id, int(json.loads(line).get("id", -1)))
            except:
                continue
    return max_id + 1


def run_conversion():

    os.makedirs(output_image_dir, exist_ok=True)
    current_id = get_start_id(output_jsonl_file)
    print(f"Current ID: {current_id}")

    all_items = []
    with open(input_jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_items.append(json.loads(line))

    total_available = len(all_items)
    num_to_sample = min(TOTAL_SAMPLES_NEEDED, total_available)

    sampled_items = random.sample(all_items, num_to_sample)

    final_records = []
    for i, item in enumerate(sampled_items):

        src_img_rel_path = item.get('image', '')
        src_img_path = osp.join(image_source_root, src_img_rel_path)
        image_filename = f"{current_id}.jpg"
        target_img_path = osp.join(output_image_dir, image_filename)

        try:
            if not osp.exists(src_img_path):
                continue
            shutil.copyfile(src_img_path, target_img_path)
        except Exception as e:
            print(f"Error occurred while copying image: {src_img_path}, Error: {e}")
            continue

        raw_prompt = item.get('prompt', '')
        raw_response = item.get('response', '')

        record = {
            "id": current_id,
            "image": image_filename,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{raw_prompt}"
                },
                {
                    "from": "gpt",
                    "value": raw_response
                }
            ]
        }
        final_records.append(record)
        current_id += 1

        if (i + 1) % 500 == 0:
            print(f"{i + 1} / {num_to_sample}...")

    with open(output_jsonl_file, 'a', encoding='utf-8') as f:
        for entry in final_records:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nprocessed {len(final_records)} items.")


if __name__ == "__main__":
    run_conversion()