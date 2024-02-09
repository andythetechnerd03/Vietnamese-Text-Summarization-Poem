import argparse
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams 



def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--model', type=str, default='Viet-Mistral/Vistral-7B-Chat',
                        help='')
    parser.add_argument('--input_path', type=str, default='/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_split_sentence.json',
                        help='')
    parser.add_argument('--output_directory', type=str, default='/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed',
                        help='')
    parser.add_argument('--num_proc', type=int, default=2,
                        help='')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='')
    args = parser.parse_args()
    return args


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path, file_save):
    with open(path, 'w') as f:
        return json.dump(file_save, f, ensure_ascii=False)


class CustomImageDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        text = self.datasets[idx]["input_ids"]
        return text

template = [
    {
        "role": "system",
        "content": "Bạn là một nhà kể chuyện phiếm, nhiệm vụ của bạn là hãy kể 1 câu chuyện đơn giản và ngắn gọn từ một bài thơ, câu chuyện nên là 1 bài liền mạch, thực tế"
    }
]


def main():
    args = parse_arguments()
    data = load_dataset("json", data_files=args.input_path, split="train")

    llm = LLM(model=args.model)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map="auto")


    def apply_template(data):
        poem = data["poem"]
        data["prompts_vllm"] = tokenizer.apply_chat_template(template + [{"role": "user", "content": poem}], tokenize=False)
        return data
    
    prompts = data.map(apply_template, num_proc=args.num_proc)


    count_file = 1    
    for idx in tqdm(range(0, len(prompts), args.batch_size)):
        final_data = []
        if idx + args.batch_size >= len(prompts):
            end = len(prompts)
        else:
            end = idx+args.batch_size
        outputs = llm.generate(prompts[idx:end]["prompts_vllm"], sampling_params)

        for output, prompt_origin in zip(outputs, prompts[idx:end]["prompts_vllm"]):
            final_data.append({"text": output.outputs[0].text.strip(), "prompt": prompt_origin})


        save_json(os.path.join(args.output_directory, f"file_{count_file}.json"), final_data)
        count_file += 1


if __name__ == "__main__":
    main()
