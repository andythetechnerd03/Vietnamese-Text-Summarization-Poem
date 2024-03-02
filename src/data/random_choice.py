import pandas as pd
import random
from joblib import Parallel, delayed
import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--input_path', type=str, default='/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/raw/5_chu_dataset.csv',
                        help='')
    parser.add_argument('--name_col_content', type=str, default='content',
                        help='')
    parser.add_argument('--output_path', type=str, default='/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_fpt.json',
                        help='')
    args = parser.parse_args()
    return args



def save_json(path, file_save):
    with open(path, 'w') as f: return json.dump(file_save, f, ensure_ascii=False)


def random_choice_sentence(poem: str):
    if type(poem) == float:
        print(poem)
    poem_split = poem.replace(" <\n> ", "\n").replace("<\n> ", "\n").split("\n\n")
    start, end = 0, len(poem_split)
    temp_poem = []
    while start < len(poem_split):
        num = random.choice(range(1, 4))
        if start + num > end:
            temp_num = end
        else:
            temp_num = start + num
        temp_poem.append("\n\n".join(poem_split[start:temp_num]))
        start = temp_num
    return temp_poem


def main():
    args = parse_arguments()
    data = pd.read_csv(args.input_path)
    poems = list(data[args.name_col_content].dropna(how='all'))

    poem_data = Parallel(n_jobs=3)(delayed(random_choice_sentence)(poem.replace("thơ năm chữ: ", "").strip()) for poem in poems)
    
    idx = 0
    data_final = []
    for poem_temp in poem_data:
        for poem_temp_temp in poem_temp:
            data_temp = {
                "id": idx,
                "poem": poem_temp_temp

            }
            data_final.append(data_temp)
            idx += 1
    save_json(args.output_path, data_final)
    

if __name__ == "__main__":
    main()
