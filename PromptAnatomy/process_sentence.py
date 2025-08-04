import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import re

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def save_to_json(data, file_path):
    json_data = data.to_dict('records')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

def split_sentences(text):
    # sentences = re.split(r'(?<=\.)\s+', text)
    sentences = re.split(r'(?<=\.)(?=\S)|(?<=\.)\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

df = read_json_file("")

required_columns = ['context']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in JSON file")

count = 0
df_1 = df.copy()
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing requests"):
    contexts = row['context']
    df_1.loc[count, 'new_context'] = contexts
    new_contexts = split_sentences(contexts)
    length = len(new_contexts)
    for i in range(length): 
        df_1.loc[count, 'contextblock_{}'.format(i)] = new_contexts[i]
    count += 1

save_to_json(df_1, "")


