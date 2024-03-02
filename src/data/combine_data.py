import json
import glob

data_files = ["/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_summarized.json", "/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_summarized_2.json"]


def combine_json_files(output_filename):
    combined_data = []
    id_counter = 0

    # Modify this glob pattern to match your file naming convention
    # for file_name in glob.glob('/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/splited_random/*.json'):
    for file_name in data_files:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # Reset the index and append to the combined list
                item['idx'] = id_counter
                combined_data.append(item)
                id_counter += 1

    # Save the combined list to a new JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

# Call the function with the desired output file name
combine_json_files('/home/pphuc/Coding/Project/Vietnamese-Poem-Summarization/data/processed/data_summarized_full.json')
