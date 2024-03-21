import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import save_json

logging.basicConfig(level=logging.INFO)


template = [
    {
        "role": "system",
        "content": "Bạn là một nhà kể chuyện phiếm, nhiệm vụ của bạn là hãy kể 1 câu chuyện đơn giản và ngắn gọn từ một bài thơ, câu chuyện nên là 1 bài liền mạch, thực tế"
    }
]


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Script for generating summaries from Vietnamese poems')
    parser.add_argument('--model', type=str, default='Viet-Mistral/Vistral-7B-Chat',
                        help='Pretrained model to use for summarization')
    parser.add_argument('--input_path', type=str, default='None',
                        help='Path to the input data file')
    parser.add_argument('--output_directory', type=str, default='None',
                        help='Directory to save the output files')
    parser.add_argument('--num_proc', type=int, default=2,
                        help='Number of processes to use for data loading')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size for the model inference')
    return parser.parse_args()


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

    for idx in tqdm(range(0, len(prompts), args.batch_size)):
        end = min(idx + args.batch_size, len(prompts))
        outputs = llm.generate(prompts[idx:end]["prompts_vllm"], sampling_params)

        final_data = [{"text": output.outputs[0].text.strip(), "prompt": prompt_origin} 
                      for output, prompt_origin in zip(outputs, prompts[idx:end]["prompts_vllm"])]

        save_json(Path(args.output_directory) / f"file_{idx // args.batch_size + 1}.json", final_data)


if __name__ == "__main__":
    logging.info('Starting script...')
    main()
    logging.info('Script finished successfully.')
