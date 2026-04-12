import json
import random
import base64
import os
import os.path as osp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict


INPUT_FILE = 'parquet_for_training' ## including win/loss image bytes and captions, generated from get_detailed_raw_train_data.py
OUTPUT_FILE = 'output_file'
IMAGE_ROOT = 'image_root'
MAX_SAMPLES = 10000
CATEGORY_TARGET_RATIO = {
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
CATEGORY_TARGET_COUNTS = defaultdict(int)

RATIO = MAX_SAMPLES / 100000

def read_image_as_bytes(image_path):
    with open(image_path, 'rb') as f:
        return f.read()

PROMPTS_TEMP = [
    "Generate an edited version of the image described by this prompt: {prompt}. The current image has noticeable flaws and low quality—please fix the errors and refine it to a clean, high-quality result.",
    "Using the prompt {prompt}, improve the existing image. It currently contains defects and looks low-quality; correct artifacts, polish details, and enhance overall visual fidelity.",
    "Create an improved edit based on {prompt}. The provided image is imperfect (artifacts, distortions, low resolution). Repair these issues and produce a sharper, more visually pleasing image.",
    "Edit the image according to {prompt}. Since the current output is flawed and not well-rendered, remove imperfections, restore missing details, and deliver a refined, high-quality version.",
    "Given the prompt {prompt}, perform image enhancement on the current result. It suffers from quality problems and visible mistakes—clean it up, correct errors, and make it look professionally finished.",
    "Please refine the image generated from {prompt}. The current image has defects (noise, artifacts, inaccurate details) and needs polishing; improve clarity, correctness, and overall quality.",
    "Based on {prompt}, regenerate or edit the current image to a higher standard. The existing image is low quality and contains visual issues; fix them and produce a crisp, clean, refined output.",
    "Improve the image corresponding to {prompt}. The current version is low quality and contains visible errors; remove artifacts, correct details, and enhance sharpness and realism.",
    "Take the image implied by {prompt} and refine the current output. Fix rendering mistakes, reduce noise, and upgrade the overall quality while keeping the intended content consistent.",
    "Edit the existing image for {prompt} to eliminate defects. Clean up artifacts, correct distortions, and produce a clearer, more polished result.",
    "Using {prompt} as the reference, enhance the current image. It has flaws and degraded quality—repair issues, improve detail, and deliver a high-fidelity version.",
    "Refine the current image generated from {prompt}. Address quality problems (blur, artifacts, incorrect textures), and produce a clean, high-quality final image.",
]

def extract_score(response):
    try:
        clean_res = response.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_res)
        return data.get('score')
    except Exception:
        return None


def read_image_as_base64(image_name):

    path = osp.join(IMAGE_ROOT, image_name)
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        return None


def process_parquet():
    skipped_low_score = 0
    skipped_parse_error = 0
    image_bytes_error = 0
    results = []

    print(f"Loading data from {INPUT_FILE}...")

    data_df = pd.read_parquet(INPUT_FILE)

    for _, data in data_df.iterrows():
        if len(results) >= MAX_SAMPLES:
            break

        max_score = data['max_score']
        if max_score is None:
            skipped_parse_error += 1
            continue

        major_category = data['major_category']
        if major_category not in CATEGORY_TARGET_RATIO:
            continue


        max_score = data.get('max_score', '')
        min_score = data.get('min_score', '')

        if not (max_score > 8 and max_score - min_score >= 3):
            skipped_low_score += 1
            continue
        CATEGORY_TARGET_COUNTS[major_category] += 1
        if CATEGORY_TARGET_COUNTS[major_category] > CATEGORY_TARGET_RATIO[major_category] * RATIO:
            continue
        prompt_text = json.loads(data['captions'])['caption']

        win_bytes = data['image']
        loss_bytes = data['image_loss']
        if not (win_bytes and loss_bytes and isinstance(win_bytes, bytes) and isinstance(loss_bytes, bytes)):
            image_bytes_error += 1
            continue
        instruction = [random.choice(PROMPTS_TEMP).format(prompt=prompt_text)]

        results.append({
            "image_win": win_bytes,
            "image_loss": loss_bytes,
            'instruction_list': [instruction]
        })
        
        
    if results:

        df = pd.DataFrame(results)
        os.makedirs(osp.dirname(OUTPUT_FILE), exist_ok=True)

        output_path = OUTPUT_FILE.replace(".parquet", f'_r{len(df)}.parquet')
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Successfully saved {len(results)} samples to {output_path}")
        print(f'skip low score {skipped_low_score}')
        print(f'skip parse error {skipped_parse_error} ')
        print(f'image bytes error {image_bytes_error}')
        print("-" * 20)
        print(CATEGORY_TARGET_COUNTS)
        return df
    


if __name__ == "__main__":
    process_parquet()