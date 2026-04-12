import json
import os
import os.path as osp

PROMPT_GALLERY = {
    "anime": """You are an expert art critic and style evaluator. Your primary goal is to assess the **Style Consistency**, **Medium Fidelity**, and **Artistic Technique** of the generated image.

**Input Data:**
- **Category:** {category}
- **Prompt:** 
{prompt}
- **Generated Image:** [Image Input]

**Evaluation Instructions:**

Please analyze the image with a focus on art style. Priority #1 is Style Adherence.

### 1. Style & Medium Adherence (Primary Importance)
Identify the specific sub-category requested in the prompt and verify its characteristics:

*   **Anime / Manga:**
    *   **2D vs. 3D:** Does the image follow the 2D or 3D instruction in prompt?
    *   **Line Work:** Are the outlines clean and consistent with the anime style?
*   **Artistic Stylization (Oil, Watercolor, Sketch, etc.):**
    *   **Texture & Brushwork:**
        *   *Oil/Impressionism:* Can you see brush strokes and texture? It shouldn't look flat or digital.
        *   *Watercolor:* Are there water stains, blending edges, and paper texture?
        *   *Sketch/Ink:* Are there hatch marks and line weight variations?
    *   **Medium Logic:** Does the image look like it was created using the physical medium requested?
*   **Realistic Stylization (CGI, 3D Render, Unreal Engine):**
    *   **Materiality:** Do surfaces (metal, plastic, skin) interact with light realistically according to the rendering engine style (e.g., Ray-tracing effects)?
*   **Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.**

### 2. Composition & Subject Integrity (Secondary Importance)
*   **Subject Clarity:** Is the subject recognizable despite the heavy stylization? (e.g., In an abstract painting, can you still see the requested cat?)
*   **Aesthetics:** Is the color palette harmonious and fitting for the genre (e.g., Pastel for Shojo Anime, Dark/Gritty for Cyberpunk)?

### Scoring Guidelines (0 - 10)

*   **10 (Stylistically Perfect):** The image perfectly mimics the requested medium. An oil painting looks like real canvas; Anime looks like a screenshot from a high-end show. No other mistakes.
*   **8-9 (Strong Style):** The style is distinct and correct. Minor digital artifacts might betray that it's AI-generated, but the artistic intent is clear.
*   **6-7 (Generic):** The style is present but weak. Example: "Oil painting" looks like a digital filter over a photo. "Anime" looks like a generic western cartoon or slightly semi-realistic.
*   **4-5 (Style Mismatch):** The subject is correct, but the style is wrong. Example: Prompt asked for "Sketch," got a "Photo."
*   **0-3 (Failure):** No style applied (looks like a default photo), or the style distorts the subject so much it becomes unrecognizable noise.

**Response Format (Strict JSON)**:
{{"analysis": "Analyze the quality of generated image according to the criteria above", "score": <int between 0-10>}}
""",
    "portrait": """You are an expert evaluator for AI-generated portraits and human photography. Your primary goal is to assess the **Framing Accuracy**, **Anatomical Correctness**, and **Photorealistic Quality** of the generated image.

**Input Data:**
- **Cateogry:** {category}
- **Prompt:**
{prompt}
- **Generated Image:**
[Image Input]

**Evaluation Instructions:**

Please analyze the image with a focus on human portraiture. Priority #1 is Framing and Anatomy.

### 1. Framing & Subject Accuracy (Primary Importance)
Check if the image strictly adheres to the requested shot type and subject details:

*   **Shot Type / Framing:**
    *   **Close-up:** Does the image focus tightly on the face/shoulders? Ensure the top of the head or chin isn't awkwardly cropped unless artistic.
    *   **Half-body:** Does the frame capture the subject from the waist/thighs up?
    *   **Full-body:** Is the **entire** figure visible from head to toe? (Crucial: Check if feet are cut off or if the head is out of frame).
*   **Subject Attributes:**
    *   **Demographics:** Are the gender, age, ethnicity, and hair color exactly as described?
    *   **Attire & Accessories:** Is the specific clothing style or accessory present?
*   **Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.**

### 2. Anatomical & Biological Correctness (Critical)
Scan the image for common AI failures in human bodies:
*   **Hands & Limbs:** Count the fingers. Are there 5 fingers per hand? Are arms/legs bent naturally?
*   **Facial Features:** Are the eyes symmetrical (unless angled)? Is the gaze direction correct? Are teeth rendered naturally (not too many/sharp)?
*   **Skin Texture:** Does the skin look realistic (pores, imperfections) or overly smooth/plastic/waxy?

### 3. Aesthetic Quality (Secondary Importance)
*   **Lighting:** Is the lighting flattering (e.g., Rembrandt, Butterfly) or consistent with the environment?
*   **Bokeh/Depth:** Is the background blur (if requested) handled naturally without "bleeding" into the hair?

### Scoring Guidelines (0 - 10)

*   **10 (Masterpiece):** Perfect framing (e.g., full body shows feet), flawless anatomy (perfect hands), and skin texture that looks indistinguishable from a real photo.
*   **8-9 (Excellent):** Correct framing and subject. Anatomy is good, maybe one finger is slightly stiff but not deformed. Skin texture is high quality.
*   **6-7 (Average):** Framing is roughly correct (e.g., Full body cuts off feet slightly). Face is good, but hands might be hidden or slightly blurry to mask errors. Skin looks a bit "AI-smooth."
*   **4-5 (Flawed):** Clear anatomical failure. Extra fingers, cross-eyed, or significant "plastic skin" look. Wrong framing (e.g., Prompt asked for Close-up, got Full-body).
*   **0-3 (Horror/Fail):** Severe deformation (extra limbs, melted face), recognizable only as a human-like shape.

**Response Format (Strict JSON)**:
{{"analysis": "Analyze the quality of generated image according to the criteria above", "score": <int between 0-10>}}
""",
    "culture": """You are an expert evaluator for cultural and historical image generation. Your primary goal is to assess the **Cultural Authenticity**, **Historical Accuracy**, and **Identity Likeness** of the generated image.

**Input Data:**
- **Category:** {category}
- **Prompt:**
{prompt}
- **Generated Image:**
[Image Input]

**Evaluation Instructions:**

Please analyze the image with a focus on cultural knowledge. Priority #1 is Authenticity and Knowledge Correctness.

### 1. Cultural & Historical Accuracy (Primary Importance)
Check the image against the specific cultural category mentioned in the prompt:

*   **Identity (Celebrity / Historical Figure):**
    *   **Likeness:** Does the face and body strictly resemble the specific person requested?
    *   **Anachronism Check:** Is the figure placed in the correct time period? Ensure no modern objects appear in historical settings (unless requested).
*   **Culture & Tradition (Festival, Religion, Art):**
    *   **Cultural Specificity:** Does the image correctly distinguish between specific cultures? (e.g., differentiate between Chinese Hanfu, Japanese Kimono, and Korean Hanbok). Avoid cultural mixing.
    *   **Symbols & Rituals:** Are religious symbols, festive decorations, and art styles (e.g., Ukiyo-e vs. Ink Wash) depicted accurately?
*   **Human Activity (Sports, Craft):**
    *   **Technical Correctness:** In sports, is the posture professional and valid? Is the equipment held correctly?
    *   **Process:** In crafts, do the tools and materials match the activity described?
*   **Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.**

### 2. General Image Quality (Secondary Importance)
*   **Aesthetics:** Color harmony, style consistency, and artistic appeal.
*   **Face/Body Quality:** Are faces rendered clearly without distortion? (Crucial for human subjects).

### Scoring Guidelines (0 - 10)

*   **10 (Authentic):** Culturally and historically perfect. The celebrity is instantly recognizable, and costumes/settings are period-accurate.
*   **8-9 (Good):** High accuracy. The person/culture is correct, with only very minor details missing (e.g., a button on a uniform).
*   **6-7 (Stereotypical/Average):** Recognizable, but relies on generic stereotypes rather than specific cultural accuracy. Or slight facial distortion.
*   **4-5 (Inaccurate):** Cultural confusion or mistaken identity. Examples: Wrong historical era clothing, mixing different religions, or the person looks like someone else.
*   **0-3 (Offensive/Fail):** Severe hallucination. Completely wrong culture, distorted human features, or offensive misrepresentation.

**Response Format (Strict JSON)**:
{{"analysis": "Analyze the quality of generated image according to the criteria above", "score": <int between 0-10>}}
""",
    "relation": """You are an expert evaluator for text-to-image generation models. Your task is to assess the quality of an image generated based on a specific text prompt. You must evaluate the image based on the input prompt provided below.

**Input Data:**
- **Category:** {category}
- **Prompt:** 
{prompt}
- **Generated Image:** 
[Image Input]

**Evaluation Instructions:**

Please analyze the image following the criteria below, ordered by importance. **Priority #1 is Prompt Adherence & Logical Correctness.**

### 1. Prompt Adherence & Logical Correctness (High Importance)
Check if the image accurately reflects the prompt's instructions. You must specifically evaluate the following dimensions based on the complexity of the prompt:

*   **Object Presence & Count:** 
    *   **objects presence:** Are all requested objects present? Are there any missing or hallucinated objects?
    *   **Count**: Does the the number of objects generated match the prompt?
*   **Attributes (Visual Properties):** 
    *   **Shape & Geometry:** Are the shapes correct?
    *   **Color:** Are the specific colors applied to the correct objects?
    *   **Texture:** Is the material or surface texture depicted accurately?
    *   **Anatomical Correctness:** Are humans/animals rendered correctly? Check for distorted faces, extra/missing fingers, or unnatural limbs.
*   **Object Relations:**
    *   **Action & Interaction:** Are the movements, poses, and physical interactions between objects natural and consistent with the prompt?
    *   **Comparison & Differentiation:** Did the model correctly distinguish between objects (e.g., "A is bigger than B", "Cat is blue, Dog is red")? Does it clearly differentiate contrasting elements?
    *   **Negation:** Did the generated image avoid elements that the prompt explicitly forbade (e.g., "no trees")?
    *   **Attribute Inference:** Are the implied details logical (e.g., if it is raining, is the ground wet)?
*   **Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.**

### 2. Image Quality & Aesthetics (Secondary Importance)
Evaluate the technical and artistic quality of the generated image:

*   **Visual Fidelity:** Is the image sharp and clear? Are there strange artifacts, noise, or blurring?
*   **Lighting & Composition:** Is the lighting realistic or stylistically appropriate? Is the composition balanced?
*   **Style Consistency:** Does the artistic style (e.g., photorealistic, oil painting, sketch) match what was requested?

---

### Scoring Guidelines (0 - 10)

Based on your analysis, assign a score according to these tiers:

*   **10 (Perfect):** The image strictly follows **all** instructions in the prompt (including complex constraints like negation, comparison, and specific textures) and has flawless visual quality.
*   **8-9 (Good):** The image completely aligns with the prompt content and logic, but falls slightly short on artistic flair or has negligible visual imperfections.
*   **6-7 (Average):** The image follows the main idea, but misses minor details (e.g., wrong texture nuance) or has moderate quality issues (e.g., slight background distortion).
*   **4-5 (Poor):** Significant failure to follow the prompt. Key objects are missing, attributes are wrong (e.g., wrong color), logical constraints (comparison/negation) are ignored, or anatomy is noticeably bad.
*   **0-3 (Very Bad):** The image is irrelevant to the prompt, contains severe hallucinations, or is visually broken/unrecognizable.

**Response Format (Strict JSON)**:
{{"analysis": "Analyze the quality of generated image according to the criteria above", "score": <int between 0-10>}}
""",
    "numeracy": """You are an expert judge specializing in evaluating the numeracy and counting capabilities of text-to-image generation models. 

Your primary task is to verify if the number of objects in the generated image strictly matches the number specified in the prompt.

**Input Data:**
- **Category:** {category}
- **Prompt:** 
{prompt}
- **Generated Image:** 
[Image Input]

Evaluation Steps:
1. **Identify the Target:** Identify the object(s) and the specific quantity requested in the prompt (e.g., "5 apples").
2. **Count Visible Objects:** Carefully count the target objects in the image. Be mindful of overlapping objects, small details, or objects blending into the background.
3. **Compare:** Compare your count with the requested count.
4. **Assess Quality (Secondary):** Only if the count is correct, assess the visual quality (fidelity, consistency, no artifacts).
5  **Other:** Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.

Scoring Rubric (Strict adherence required):

*   **Score 0-2 (Critical Failure):** The count is incorrect. Even if the image is beautiful, if the number is wrong, the score must be in this range.
*   **Score 3-4 (Ambiguous/Major Flaws):** The count seems correct but objects are heavily distorted, merged, or extremely difficult to distinguish (e.g., a "six-fingered hand" when asking for 5 fingers, or a hydra-like object).
*   **Score 5-7 (Accurate but Low Quality):** The count is exactly correct, but the image suffers from low resolution, bad anatomy, or poor aesthetic quality.
*   **Score 8-9 (Accurate and Good Quality):** The count is exactly correct, and the image is clear, logical, and visually pleasing.
*   **Score 10 (Perfect):** Flawless execution. The count is exact, distinct, and the image is of high artistic or photorealistic quality with no artifacts.

Response format (JSON):
{{"target_quantity": "The number requested", "detected_quantity": "The number you counted", "analysis": "Brief reasoning focused on counting and visibility", "score": a score in 0-10}}
""",
    "science": """You are an expert evaluator for image generation. Your primary goal is to assess the **Natural Science Knowledge** of the generated image based on the prompt.

**Input Data:**
- **Category:** {category}
- **Prompt:** 
{prompt}
- **Generated Image:** 
[Image Input]

**Evaluation Instructions:**

Please analyze the image with a focus on scientific validity mentioned in the prompt. Priority #1 is Factual Accuracy.

### 1. Scientific Accuracy & Logic (Primary Importance)
Check the image against the specific scientific domain if it is mentioned **in the prompt**:

*   **Biology (Plants & Animals):**
    *   **Species Fidelity:** Does the plant/animal look like the specific species requested? (e.g., Is the leaf shape correct? Does the tiger have the right pattern?)
    *   **Anatomy:** Check for biological hallucinations. Are there extra limbs, distorted wings, or unnatural joints, and so on?
    *   **Habitat:** Is the environment ecologically consistent with the organism?
*   **Physics & Chemistry:**
    *   **Physical Laws:** Does the image obey gravity and optics? Check reflections (mirrors/water), shadows (direction/softness), and fluid dynamics.
    *   **Apparatus & States:** Is the image correct in the Chemistry?
*   **Earth Science / Real-world Phenomena:**
    *   **Geology & Weather:** Are natural features (mountains, volcanoes, lightning, clouds) rendered with realistic textures and physical behavior?
*   **Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.**

### 2. General Image Quality (Secondary Importance)
*   **Visual Fidelity:** Sharpness, resolution, and absence of AI artifacts (noise, blurring).
*   **Composition:** Is the main subject clearly visible and well-framed?

### Scoring Guidelines (0 - 10)

*   **10 (Scientifically Accurate):** Flawless representation of the subject. Anatomy is perfect, lighting/physics are realistic.
*   **8-9 (Good):** Correct in Knowledge. Minor stylistic issues, but no violation of physical laws or biological structure.
*   **6-7 (Average):** Recognizable, but contains minor inaccuracies (e.g., slight anatomical awkwardness, shadow direction is a bit off).
*   **4-5 (Unrealistic):** Significant scientific errors. Examples: Animals with extra legs, water flowing upwards, objects floating without reason.
*   **0-3 (Fail):** Complete hallucination. The object is unrecognizable or violates basic reality (e.g., a cat that looks like a car).

**Response Format (Strict JSON)**:
{{"analysis": "Analyze the quality of generated image according to the criteria above", "score": <int between 0-10>}}
""",
    "spatio": """You are an expert evaluator for spatial reasoning in computer vision. Your primary goal is to assess the **Spatial Accuracy**, **Geometric Logic**, and **Perspective Consistency** of the generated image based on the prompt.

**Input Data:**
- **Category:** {category}
- **Prompt:**
{prompt}
- **Generated Image:**
[Image Input]

**Evaluation Instructions:**

Please analyze the image with a focus on spatial relations. Priority #1 is Spatial Logic.

### 1. Spatial & Geometric Logic (Primary Importance)
Analyze whether the model correctly understood the spatial instructions in the prompt:

*   **Spatial Relations (Left/Right/Up/Down):**
    *   **Relative Position:** Are objects placed exactly where requested? (e.g., "A is to the left of B"). Check strictly for Left/Right reversals.
    *   **Proximity:** Are objects "near," "far," or "touching" as described?
*   **Occlusion Reasoning (Hiding/Behind):**
    *   **Layering:** If Object A is "behind" Object B, is A correctly partially hidden? Ensure there are no "transparency glitches" (seeing lines through solid objects) or incorrect overlapping.
    *   **Depth Cues:** Does the image clearly imply which object is in the foreground and which is in the background?
*   **Perspective & Views:**
    *   **Viewpoint:** Is the camera angle correct? (e.g., "Top-down view," "Worm's eye view," "Isometric view").
    *   **Vanishing Points:** Do parallel lines converge correctly? Check for distorted architecture or objects that look "flat" (2D) when they should be volumetric (3D).
    *   **Scale Consistency:** Do distant objects appear smaller than near objects in a logical way?
*   **Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.**

### 2. Image Quality & Composition (Secondary Importance)
*   **Distortion:** Are straight lines actually straight? Are geometric shapes (circles, squares) rendered correctly?
*   **Clarity:** Is the spatial arrangement easy to read, or is it a messy clutter?

### Scoring Guidelines (0 - 10)

*   **10 (Perfect Spatial Logic):** All positional constraints (Left/Right/Behind) are perfectly executed. Perspective is realistic and depth is convincing. No other mistakes.
*   **8-9 (Good):** Spatially correct. Minor perspective flaws that don't affect the prompt's instructions.
*   **6-7 (Ambiguous):** The layout is roughly correct, but exact relations are messy (e.g., A is "sort of" next to B, but overlapping strangely).
*   **4-5 (Spatial Error):** Clear failure in spatial instructions. Example: "A on the Left" is generated on the Right. "Top-down view" looks like a side view.
*   **0-3 (Geometric Failure):** Impossible geometry (Escher-like distortions), severe occlusion failures (objects merging into each other), or total disregard for layout.

**Response Format (Strict JSON)**:
{{"analysis": "Analyze the quality of generated image according to the criteria above", "score": <int between 0-10>}}
""",
    "temporal": """You are an expert evaluator for temporal and contextual reasoning in images. Your primary goal is to assess the **Chronological Consistency**, **Age Progression**, and **Environmental Logic** of the generated image.

**Input Data:**
- **Category:** {category}
- **Prompt:** 
{prompt}
- **Generated Image:** [Image Input]

**Evaluation Instructions:**

Please analyze the image with a focus on time-based cues. Priority #1 is Temporal Consistency.

### 1. Temporal Reasoning & Context (Primary Importance)
Analyze whether the image accurately depicts the specific time, age, or sequence requested:

*   **Horizontal & Longitudinal Time:**
    *   **Sequence (Before/After):** If the prompt implies a state change (e.g., "a broken vase," "a melting ice cream"), does the image capture the correct stage of the event?
    *   **Age Progression:** Do human or animal subjects look the specific age requested? (e.g., wrinkles for old age, proportions for toddlers). Do objects look "weathered," "ancient," or "futuristic" as requested?
*   **Temporal Context (Time of Day / Season):**
    *   **Lighting & Atmosphere:** Does the lighting match the time? (e.g., Long shadows for "sunset," dark blue/black sky for "midnight").
    *   **Seasonal Indicators:** Are the flora and weather consistent with the season? (e.g., No green leaves in "dead of winter," flowers blooming in "spring").
*   **Geographical & Seasonal Logic:**
    *   **Location-Time Consistency:** If a location is specified along with a time, is it logical? (e.g., "Christmas in Australia" should look summery/hot, not snowy).
    *   **Historical Context:** If a specific era is implied (e.g., "1920s"), are the technology, fashion, and background elements historically appropriate?
*   **Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.**

### 2. Image Quality & Aesthetics (Secondary Importance)
*   **Consistency:** Ensure there are no conflicting time cues (e.g., a bright sun in a starry night sky).
*   **Visual Appeal:** Quality of lighting rendering and atmospheric effects.

### Scoring Guidelines (0 - 10)

*   **10 (Perfect Temporal Logic):** The image perfectly captures the mood, age, and specific time constraints. Logic (like Season vs. Hemisphere) is sound.
*   **8-9 (Good):** Accurate time depiction. Age and season are correct. Minor lighting inconsistencies allowed.
*   **6-7 (Average):** The general time is correct (e.g., it is night), but details are off (e.g., shadows don't match the light source, or a 50-year-old looks 30).
*   **4-5 (Temporal Conflict):** Contradictory elements. Example: Snowing while the sun is high and bright; A "baby" looks like a mini-adult; An "ancient" ruin looks brand new.
*   **0-3 (Logical Failure):** Complete failure to represent time. Night is drawn as day. Historical eras are mixed randomly.

**Response Format (Strict JSON)**:
{{"analysis": "Analyze the temporal consistency and logic of the generated image according to the criteria above", "score": <int between 0-10>}}
""",
    "text": """You are an expert AI assistant specialized in evaluating "Text Rendering" capabilities of image generation models. Your goal is to judge whether the model has accurately rendered the specific text requested in the prompt within the generated image.

Input Data:
- Category: {category}
- Generation Prompt: 
{prompt}
- Generated Image: [Image Input]

Evaluation Criteria (Prioritized):

1. Text Accuracy (Highest Priority): 
   - Does the text in the image STRICTLY match the requested text in the prompt? 
   - Check for spelling errors, missing letters, duplicated letters, or hallucinated words.
   - Case sensitivity is secondary unless explicitly requested, but the characters must be correct.
   
2. Legibility & Structure:
   - Are the glyphs (letters/characters) well-formed and distinct? 
   - Avoid "pseudo-language" (alien hieroglyphs) or blurred/melting strokes.
   - Is the text readable, or is it obscured by background elements/low contrast?
   - Is there any other mistake in the image, or not following the prompt?

3. Style & Integration:
   - Does the font style, color, and material match the prompt description?
   - Is the text naturally integrated into the scene's perspective and lighting (rendering), or does it look like a flat overlay?
4. Any other aspects that in abnormal(not following the prompt) or uncorrect in the image.

Scoring Guide (0-10):

- 10: Perfect. Text is 100% accurate, perfectly clear, and style/integration matches the prompt flawlessly, no other mistakes in the image.
- 9: Excellent. Text is 100% accurate, but minor aesthetic flaws in style or background.
- 7-8: Good. Text is accurate, but there might be very slight legibility or style issues.
- 4-6: Failed Accuracy. Text has minor typos (1-2 wrong letters), is partially cut off, or is hard to read.
- 1-3: Severe Failure. Text is incoherent, major spelling errors, missing entirely, or looks like gibberish.
- 0: Irrelevant. No text present or image is completely unrelated.

Response Format (Strict JSON):
{{"analysis": "Briefly analyze the spelling accuracy first, then the style/rendering quality.", "score": <int between 0-10>}}
""",
}

CATEGORY_MAP = {
    "General Knowledge": "culture",
    "Spatio Reasoning": "spatio",
    "Relational Operations": "relation",
    "Temporal Reasoning": "temporal",
    "Natural Science": 'science',
    "Counting": "numeracy",
    "General Object": "relation",
    "Text Rendering": "text",
    "Portrait": "portrait",
    "Stylization": "anime"
}

def dump_reward_jsonl(generation_prompt_path, out_path, num_imgs=8):
    with open(generation_prompt_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    
    new_data_list = []
    for idx, line in enumerate(lines):
        try:
            data = json.loads(line)
            category = CATEGORY_MAP[data['major_category']]
        except json.decoder.JSONDecodeError as e:
            print(f"wrong message: {e}")
            print(f"wrong content: {line[:500]!r}")
            raise RuntimeError
        if data['subcategory'] == "TIIF":
            subcategory = "text"
        else:
            subcategory = data['subcategory']
        reward_prompt = PROMPT_GALLERY[category].format(prompt=data['prompt'], category=subcategory)
        for img_idx in range(num_imgs):
            img_folder = str(idx).zfill(5)
            img_name = str(img_idx).zfill(5)
            img_path = osp.join(img_folder, "samples", img_name + ".png")

            new_data = {'major_category': data['major_category'], "subcategory": data['subcategory'], 'prompt': reward_prompt, 'image': img_path, 'gen_prompt': data['prompt']}
            new_data_list.append(new_data) 
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in new_data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    file_path = "prompts_example.jsonl"
    out_path = "reward_prompts_example.jsonl"
    dump_reward_jsonl(file_path, out_path)
