import pandas as pd
import json
import os

file_path = "origin_train.parquet"
save_path = file_path.replace(".parquet", "_cleaned.parquet")

print(f"processing: {file_path}")
df = pd.read_parquet(file_path)
original_count = len(df)
print(f"raw lines: {original_count}")

def is_valid_json_row(json_str):
    try:
        data = json.loads(json_str)
        
        if not isinstance(data, dict):
            return False

        for key, value in data.items():
            if value is None:
                return False
            if not isinstance(value, str):
                return False
            if isinstance(value, str) and value.strip() == "":
                return False
                
        return True
    except Exception:
        return False

def is_valid_bytes(img_bytes):
    return isinstance(img_bytes, bytes)

bad_rows = df[~df['captions'].apply(is_valid_json_row)]
if len(bad_rows) > 0:
    print("dirty row::")
    print(bad_rows.iloc[0]['captions'])

df_clean = df[df['captions'].apply(is_valid_json_row)]
df_clean = df_clean[df_clean['image'].apply(is_valid_bytes)]


cleaned_count = len(df_clean)
removed_count = original_count - cleaned_count

print("-" * 30)
print(f"cleaned lines: {cleaned_count}")
print(f"removed lines: {removed_count}")

if removed_count > 0:
    print(f"Saving cleaned data to: {save_path}")
    df_clean.to_parquet(save_path, engine='pyarrow', index=False)
    print("Cleaned data saved successfully.")
else:
    print("No invalid rows found. No need to save.")

if cleaned_count > 0:
    print("-" * 30)
    print("First cleaned row captions:")
    print(df_clean.iloc[0]['captions'])
