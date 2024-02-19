# # '<s>[INST] <<SYS>>\nBạn là một nhà kể chuyện chuyên nghiệp, nhiệm vụ của bạn là hãy kể 1 câu chuyện từ một bài thơ, câu chuyện nên là 1 bài liền mạch kể về câu chuyện gì đó\n<</SYS>>\n\nnhững con đường tôi đi\nngày ấy vàng hoa điệp\nrộn rã và thầm thì\ntình nhân khoe áo đẹp\n\nngày ấy biển rất xanh\nlá bàng thì rất đỏ\ncao to là bóng anh\nbóng tôi thì bé nhỏ\n\nđêm ngã dài trên cát\nnhững chòm sao rất xa\nvăng vẳng tiếng ai hát\ntôi họa theo ngâm nga [/INST] Ngày xưa, khi tôi còn là một đứa trẻ, những con đường tôi đi đầy ắp những kỷ niệm. Những con đường đó được tô điểm bởi những bông hoa điệp vàng, chúng nhảy múa và thì thầm trong gió. Tôi nhớ mình đã nhìn thấy những người yêu nhau, tay trong tay, mặc những bộ quần áo đẹp nhất của họ.\n\nBiển ngày ấy xanh thẳm, và lá bàng đỏ rực. Tôi thường đi cùng anh, bóng của anh cao lớn và bao trùm lấy tôi. Chúng tôi cùng nhau đi dọc bờ biển, lắng nghe tiếng sóng vỗ vào bờ.\n\nKhi màn đêm buông xuống, chúng tôi nằm trên cát, ngắm nhìn những vì sao xa xăm. Tiếng hát của ai đó vang vọng trong không khí, và tôi đã vẽ nên những giai điệu đó trong tâm trí mình.\n\nNhững ký ức này đã ở lại với tôi suốt cuộc đời, nhắc nhở tôi về một thời đơn giản hơn, khi thế giới dường như nhỏ bé hơn và mọi thứ đều có thể.</s>'


# import json

# def load_json(path):
#     with open(path, 'r') as f: return json.load(f)

# def save_json(path, file_save):
#     with open(path, 'w') as f: return json.dump(file_save, f, ensure_ascii=False, indent=4)


# dataset = load_json("/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_summarized.json")

# # this function is used to output the right formate for each row in the dataset
# def create_text_row(instruction, output, input):
#     text_row = f"""<s>[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\n{input} [/INST] {output}</s>"""
#     return text_row

# instruct_prompt = "Bạn là một nhà thơ chuyên nghiệp, nhiệm vụ của bạn là chuyển bài văn này thành 1 bài thơ 5 chữ từ khoảng 1 đến 3 khổ"
# final_data = []
# # interate over all the rows formate the dataset and store it in a jsonl file
# def process_json_file(output_file_path):
#     with open(output_file_path, "w") as output_jsonl_file:
#         for item in dataset:
#             json_object = {
#                 "text": create_text_row(instruct_prompt, item["summarized"], item["prompt"]),
#                 "instruction": instruct_prompt,
#                 "input": item["summarized"],
#                 "output": item["prompt"]
#             }
#             # output_jsonl_file.write(json.dumps(json_object, ensure_ascii=False) + "\\n")
#             final_data.append(json_object)
#         save_json(output_file_path, final_data)

# # Provide the path where you want to save the formatted dataset
# process_json_file("/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/training_dataset.json")

from datasets import load_dataset

data = load_dataset("json", data_files="/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/training_dataset.json", cache_dir=".")

data.push_to_hub("pphuc25/poem-5-words-vietnamese", private=True)