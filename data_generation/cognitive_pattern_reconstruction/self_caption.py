import random
import pandas as pd
import json
import os
from collections import defaultdict

parquet_file = 'parquet_for_training.parquet'
output_image_dir = 'output_image_folder'
output_jsonl_file = 'output_jsonl_path.jsonl'
sample_num = 10000
RATIO = sample_num / 100000


# human_prompts = [
#     "<image>\nDescribe the image in detail.",
#     "<image>\nProvide a detailed description of the given image.",
#     "<image>\nWhat is shown in this image? Give a comprehensive explanation.",
#     "<image>\nAnalyze the image and describe its contents thoroughly.",
#     "<image>\nPlease provide a long and detailed caption for this picture."
# ]
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


prompt_template_gallery = {
    "General Object": [
    "Guess the original prompt that generated this image. Focus on what objects are present (single or multiple), plus visual attributes like shape, color, texture, and whether it looks like a real-world photo. Output only the prompt.",
    "Infer the image-generation prompt from the picture. Describe the main object(s) and any key properties—form, palette, surface detail/material, and realism level. Return exactly one prompt.",
    "Reverse-engineer the prompt behind this image. Treat it like an object-centric description: identify the subject(s), how many there are, and standout cues such as geometry, colors, and textures. Provide only the predicted prompt text.",
    "Recover the most likely prompt used for this image. Use the scene’s object details as evidence—materials (metal/wood/fabric), surface texture, dominant colors, and whether it’s a clean render or a real-world style shot. Output one prompt.",
    "Predict the prompt that produced this image by summarizing the visible object(s) and their appearance (shape, color scheme, texture, real-world vs stylized). Answer with a single concise prompt."
    ],
    "Relational Operations" : [
    "Guess the original prompt used to generate this image. Look for how two or more things relate—someone doing something to something else, a larger/smaller or more/less comparison, telling items apart, or an explicit 'not' statement. Output only the predicted prompt text.",
    "Infer the image generation prompt from the image. The answer is probably expressed through relationships: interaction between subjects, contrast or comparison, distinguishing features, or negation. Reply with a single prompt.",
    "Reconstruct the prompt that produced this image. Pay attention to relational wording you’d use to describe it (e.g., one thing vs another, same vs different, which one is excluded, who interacts with whom). Output one likely prompt.",
    "Recover the prompt behind this image. What matters most is the relational logic—actions/contacts between entities, comparative attributes, separation into categories, or describing what is absent/denied. Provide only the prompt.",
    "Predict the generation prompt for this image. Use the scene’s relationships as clues: who is interacting, what is being compared, what is distinct, and what is explicitly ruled out. Return one concise prompt."
    ],
    "Natural Science": [
    "Guess the original prompt used to generate this image. Use clues from natural science—plants, animals, physical phenomena, or basic chemistry. Output only the predicted prompt text.",
    "Infer the image generation prompt from the image. Think in terms of nature and science: organisms, habitats, anatomy, motion/forces, light/sound/energy, or simple chemical reactions. Reply with a single prompt.",
    "Reconstruct the prompt that likely produced this image. Describe it the way a science-themed prompt would: a plant or animal subject, or a clear physics/chemistry concept shown visually. Output one likely prompt.",
    "Recover the prompt behind this image. Focus on scientific content you can name—species or plant parts, observable behaviors, weather or physical effects, lab materials, mixtures, or reactions. Provide only the prompt.",
    "Predict the generation prompt for this image. Let the scene guide you toward a natural science description (plants/animals or a straightforward physics/chemistry idea), and return one concise prompt."
    ],
    "General Knowledge": [
    "Guess the original prompt used to generate this image. Use cultural-knowledge clues such as festivals, sports, religion, traditional crafts, or a well-known public figure. Output only the predicted prompt text.",
    "Infer the image generation prompt from the image. Consider cultural context: holiday scenes, athletic activities, religious symbols/rituals, handmade craftwork, or a famous celebrity-like subject. Reply with a single prompt.",
    "Reconstruct the prompt that likely produced this image. Describe it as a cultural scene—celebration or festival atmosphere, sport setting, spiritual elements, artisan craft details, or a prominent celebrity theme. Output one likely prompt.",
    "Recover the prompt behind this image. Focus on recognizable cultural signals: costumes, ceremonies, sports gear, sacred architecture/icons, craft techniques/materials, or fame/celebrity presentation. Provide only the prompt.",
    "Predict the generation prompt for this image. Let cultural cues guide you (festival, sports, religion, craft, celebrity), and return one concise prompt that best matches the image."
    ],
    "Spatio Reasoning": [
    "From the image alone, reconstruct the exact text prompt that could have generated it. Pay attention to spatial cues like depth, viewpoint, occlusion, and 2D/3D layout. Output only the prompt.",
    "What was the original generation prompt for this picture? Use geometry and perspective as your main clues—overlapping objects, camera angle, relative positions, and planar vs volumetric shapes. Return one prompt only.",
    "Reverse-engineer the prompt behind this image. Describe the scene in terms of spatial structure: where things are placed, what blocks what, and how the view is framed (top-down, side view, close-up, wide angle). Provide a single prompt.",
    "Predict the prompt that produced this image. Focus on spatial reasoning elements such as 2D diagrams vs 3D scenes, hidden parts due to occlusion, and changes in viewpoint or perspective. Only output the predicted prompt text.",
    "Recover the likely image-generation prompt. Let the composition guide you—foreground/background separation, vanishing lines, object overlap, and how the camera sees the scene from a particular angle. Answer with one concise prompt."
    ],
    "Temporal Reasoning": [
    "Reconstruct the original prompt that generated this image. Use time-related cues—simultaneous events in one moment or changes across a timeline. Output only the prompt text.",
    "Guess the exact generation prompt for this picture. Think about what’s happening at the same time versus what’s changing over time (before/after, stages, progression). Return a single prompt.",
    "Reverse-engineer the prompt behind the image. Describe it with temporal structure in mind: a snapshot of concurrent actions, or a sequence showing evolution, transition, or phases. Provide only one prompt.",
    "Predict the prompt that produced this image. Look for timeline signals such as step-by-step states, growth/decay, cause-and-effect over time, or multiple moments depicted together. Output only the predicted prompt.",
    "Recover the likely image prompt by focusing on temporal reasoning: what is synchronized within a time window, and what indicates a specific stage or change along time. Answer with one concise prompt."
    ],
    "Portrait": [
    "Reconstruct the original image-generation prompt for this portrait. Use cues like framing (close-up, half-body, full-body), pose, outfit, lighting, and background. Output only the prompt text.",
    "Guess what prompt produced this portrait. Consider how the subject is framed, the camera distance, expression, styling, and photographic/artistic style. Return a single prompt.",
    "Infer the generation prompt from the image. Describe the person in a portrait-style prompt: composition, viewpoint, clothing, mood, and lighting setup. Provide only one prompt.",
    "Reverse-engineer the prompt behind this portrait image. Pay attention to portrait details—shot type, depth of field, color tone, and any studio vs outdoor setting. Output only the predicted prompt.",
    "Predict the prompt used to generate this portrait by summarizing the subject and portrait composition (close/half/full body), along with key styling and atmosphere. Answer with one concise prompt."
    ],
    "Text Rendering": [
    "Guess the original prompt that generated this image. Focus on any rendered text—where it appears in the scene, its style (natural signage, poster/infographic design, handwriting/graffiti), and the overall layout. Output only the prompt.",
    "Infer the image-generation prompt from the picture. Use the typography as the main clue: font/handwriting style, placement, background context, and what the text is supposed to say. Return a single prompt.",
    "Reverse-engineer the prompt behind this image. Describe it as a text-forward scene: sign in a real environment, a designed poster/menu/infographic, or handwritten/graffiti text. Provide only one prompt.",
    "Predict the prompt used to produce this image. Pay attention to legible words, text formatting, and composition (headline/body text, alignment, spacing, colors). Output only the predicted prompt text.",
    "Recover the likely generation prompt by describing the text rendering: content of the text (if readable), the medium (sign, print, screen, marker, spray paint), and the design or scene context. Answer with one concise prompt."
    ],
    "Stylization": [
    "Reconstruct the original prompt that generated this image. Emphasize the visual style—anime look or a specific artistic rendering (oil painting, watercolor, ink, impressionist, etc.). Output only the prompt text.",
    "Guess the prompt behind this image by reading its stylization: is it anime/cel-shaded, or fine-art inspired with visible brushwork and painterly color? Return a single prompt.",
    "Infer the image-generation prompt from the picture. Describe the subject plus the dominant art style cues (linework, shading, texture, palette, medium). Provide only one prompt.",
    "Reverse-engineer the prompt that likely produced this image. Focus on the stylistic instructions someone would type: anime character art vs a named/traditional art medium and technique. Output only the predicted prompt.",
    "Predict the prompt used to generate this image, leaning on style signals such as illustration vs painting, flat colors vs brushstrokes, and overall aesthetic direction. Answer with one concise prompt."
    ],
    "Counting": [
    "Guess the original prompt that generated this image. Focus on how many items/subjects are shown and how they’re arranged. Output only the predicted prompt text.",
    "Infer the image-generation prompt from the picture. The key clue is quantity—count the objects/people/animals and describe them accordingly. Return a single prompt.",
    "Reverse-engineer the prompt behind this image by identifying the exact number of visible elements and their types (and any simple layout cues). Provide only one prompt.",
    "Predict the prompt that produced this image. Use counting as the main signal: the scene likely specifies an exact number of objects, possibly with colors or positions. Output only the prompt.",
    "Recover the likely generation prompt by writing what someone would type to control quantity (e.g., 'three X', 'five Y') while matching the scene. Answer with one concise prompt."
    ]
}

def get_next_id(file_path):

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0

    max_id = -1
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                current_id = int(data.get("id", -1))
                if current_id > max_id:
                    max_id = current_id
            except (json.JSONDecodeError, ValueError):
                continue
    return max_id + 1


if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)


current_global_id = get_next_id(output_jsonl_file)
print(f"Current global ID: {current_global_id}")

df = pd.read_parquet(parquet_file)
jsonl_records = []

for index, row in df.iterrows():

    image_filename = f"{current_global_id}.jpg"
    image_path = os.path.join(output_image_dir, image_filename)

    with open(image_path, "wb") as f:
        f.write(row['image'])

    try:
        if isinstance(row['captions'], str) and (row['captions'].startswith('{') or row['captions'].startswith('[')):
            caption_data = json.loads(row['captions'])
            caption_text = caption_data.get("caption", row['captions'])
        else:
            caption_text = row['captions']
    except Exception:
        caption_text = str(row['captions'])

    major_category = str(row['major_category'])
    CATEGORY_TARGET_COUNTS[major_category] += 1
    if CATEGORY_TARGET_COUNTS[major_category] > CATEGORY_TARGET_RATIO[major_category] * RATIO:
        continue

    selected_prompt = random.choice(prompt_template_gallery[major_category])
    selected_prompt = "<image>\n" + selected_prompt

    record = {
        "id": current_global_id,
        "image": image_filename,
        "category": major_category,
        "conversations": [
            {"from": "human", "value": selected_prompt},
            {"from": "gpt", "value": caption_text}
        ]
    }
    jsonl_records.append(record)

    current_global_id += 1

with open(output_jsonl_file, 'w', encoding='utf-8') as f:
    for entry in jsonl_records:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


print(f"process success for {len(jsonl_records)} data. Current ID: {current_global_id - 1}")
print("category counts:")
print(CATEGORY_TARGET_COUNTS)