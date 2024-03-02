import json
from glob import glob
def load_json(path):
    with open(path, 'r') as f: return json.load(f)

data_files = glob("/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/raw/summarized_data/*")
count = 0
final_data = []
for data_file in data_files:
    data = load_json(data_file)
    for data_th in data:
        summarized = data_th["text"]
        prompt = data_th["prompt"].split("\n\n")[-1].replace("[/INST]", " ")

        final_data.append({
            "idx": count,
            "prompt": prompt.strip(),
            "summarized": summarized.strip()
        })
        count += 1

import json

def save_json(path, file_save):
    with open(path, 'w') as f: return json.dump(file_save, f, ensure_ascii=False)

save_json("/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_summarized_final.json", final_data)