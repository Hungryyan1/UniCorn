import json
from collections import defaultdict
from pathlib import Path


def check_duplicate_prompts_dedup(file_path):

    prompt_to_ids = {}
    
    new_data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                raw_prompt = data.get('prompt', '')
                if isinstance(raw_prompt, (dict, list)):
                    prompt = json.dumps(raw_prompt, sort_keys=True, ensure_ascii=False)
                    print(prompt)
                else:
                    prompt = str(raw_prompt)

                item_id = data.get('id', line_num - 1)
                major_category = data.get('major_category', '')
                subcategory = data.get('subcategory', '')
                if prompt not in prompt_to_ids:
                    new_data_list.append(data)
                    prompt_to_ids[prompt] = {'ids': [], 'major_category': major_category, 'subcategory': subcategory}
                prompt_to_ids[prompt]['ids'].append(item_id)
            except json.JSONDecodeError as e:
                print(f"warning: wrong in line: {line_num} with {e}")
                continue
    
    duplicates = {}
    for prompt, info in prompt_to_ids.items():
        if len(info['ids']) > 1:

            duplicates[prompt] = {
                'ids': info['ids'],
                'major_category': info.get('major_category', ''),
                'subcategory': info.get('subcategory', '')
            }
    
    out_path = file_path.replace('.jsonl', '_dedup.jsonl')

    with open(out_path, 'w', encoding='utf-8') as out_f:
        for item in new_data_list:
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return {
        'total_prompts': len(prompt_to_ids),
        'unique_prompts': len(prompt_to_ids) - len(duplicates),
        'duplicate_prompts_count': len(duplicates),
        'duplicate_combinations': sum(len(ids) - 1 for ids in duplicates.values()),
        'duplicates': duplicates
    }


def main():
    file_path = 'input_file_path.jsonl'
    
    result = check_duplicate_prompts_dedup(file_path)
    
    print(f"total_prompts: {result['total_prompts']}")
    print(f"unique_prompts: {result['unique_prompts']}")
    print(f"duplicate_prompts_count: {result['duplicate_prompts_count']}")
    print(f"duplicate_combinations: {result['duplicate_combinations']}")
    print("-" * 80)
    
    if result['duplicates']:
        print("\nduplicates:")
        print("=" * 80)
        for idx, (prompt, ids) in enumerate(result['duplicates'].items(), 1):
            print(f"\nduplicate #{idx}:")
            print(f"  Prompt: {prompt}")
            print(f"  Count: {len(ids)}")
            print(f"  Involved IDs: {ids}")
    else:
        print("\n✓ No duplicates found")
    
    # Output statistics summary
    print("\n" + "=" * 80)
    print("summary:")
    print(f"duplicate_combinations: {result['duplicate_combinations']}")
    if result['duplicates']:
        all_duplicate_ids = []
        for info in result['duplicates'].values():
            all_duplicate_ids.extend(info['ids'])
        print(f"Involved duplicate IDs: {len(all_duplicate_ids)}")
        print(f"all duplicate IDs: {sorted(all_duplicate_ids)}")

    if result['duplicates']:
        major_counter = {}
        sub_counter = {}
        for info in result['duplicates'].values():
            major = info.get('major_category', '') or '<UNKNOWN>'
            sub = info.get('subcategory', '') or '<UNKNOWN>'
            major_counter[major] = major_counter.get(major, 0) + 1
            sub_counter[sub] = sub_counter.get(sub, 0) + 1

        print("\ncount by category （only duplicate prompts）:")
        print(f"distinct major_category types: {len(major_counter)}")
        for k, v in sorted(major_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {k}: {v}")
        print(f"distinct subcategory types: {len(sub_counter)}")
        for k, v in sorted(sub_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {k}: {v}")
    else:
        print("\nNo duplicate prompts found, skipping category-wise statistics.")

if __name__ == '__main__':
    main()

