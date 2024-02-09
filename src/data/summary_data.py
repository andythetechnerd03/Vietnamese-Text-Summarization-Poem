import argparse
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from tqdm import tqdm
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--model', type=str, default='Viet-Mistral/Vistral-7B-Chat',
                        help='')
    parser.add_argument('--input_path', type=str, default='/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_split_sentence.json',
                        help='')
    parser.add_argument('--output_path', type=str, default='/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_fpt_summary.json',
                        help='')
    args = parser.parse_args()
    return args


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path, file_save):
    with open(path, 'w') as f:
        return json.dump(file_save, f)


def main():
    args = parse_arguments()
    # data = load_json(args.input_path)
    data = load_dataset("json", args.input_path, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto", 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map="auto")

    final_data = []
    for data_th in tqdm(data):
        conversation = [{"role": "system",
                        "content": "Bạn là một nhà kể chuyện chuyên nghiệp, nhiệm vụ của bạn là hãy kể 1 câu chuyện từ một bài thơ"}]

        conversation.append({"role": "user", "content": data_th})
        # input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            temperature=0.1,
            repetition_penalty=1.05,
        )
        output = tokenizer.batch_decode(
            out_ids[:, input_ids.size(1):], skip_special_tokens=True)[0].strip()

        output_formated = {
            "input": data_th,
            "output": output
        }

        final_data.append(output_formated)

    save_json(args.ouput_path, final_data)


if __name__ == "__main__":
    main()
