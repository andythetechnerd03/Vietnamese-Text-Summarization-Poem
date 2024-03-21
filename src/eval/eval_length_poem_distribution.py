import json
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict
import sys

import plotly.express as px

def download_nltk_packages() -> None:
    """Download necessary nltk packages."""
    nltk.download('stopwords')
    nltk.download('punkt')

def load_json_file(file_path: str) -> Dict:
    """Load and return data from a JSON file."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return {}

def process_poems(data: Dict) -> List[int]:
    """Process poems and return a list of token lengths."""
    token_lengths = []
    for poem_data in data:
        lines = poem_data["poem"].split("\n")
        tokens = [word_tokenize(re.sub(r'[^\w\s]', '', line)) for line in lines]
        token_lengths += [len(t) for t in tokens]
    return token_lengths

def calculate_length_percentages(token_lengths: List[int]) -> Dict[int, float]:
    """Calculate and return length percentages."""
    length_counts = Counter(token_lengths)
    total_tokens = sum(length_counts.values())
    return {length: round(count * 100 / total_tokens, 2) for length, count in length_counts.items()}

def create_dataframe(length_percentages: Dict[int, float]) -> pd.DataFrame:
    """Create and return a DataFrame from length percentages."""
    return pd.DataFrame(length_percentages.items(), columns=['line_length', 'percentage'])

def create_plot(df: pd.DataFrame) -> None:
    """Create and display a plot from a DataFrame."""
    fig = px.bar(df, x='line_length', y='percentage',
                 title="Distribution of length in poem line with Vistral (baseline)",
                 text_auto=True)
    fig.update_layout(xaxis_title="Sentence length", yaxis_title="Probability (in percent)")
    fig.update_traces(textfont_size=14, textangle=0, textposition="outside")
    fig.update_xaxes(range=[0,11], tickvals=list(range(-1, 12)))
    fig["data"][0]["marker"]["color"] = ['#8690FF' if x == 5 else '#7FD4C1' for x in fig["data"][0]["x"]]
    fig.show()

def main(file_path: str) -> None:
    """Main function to process data and create a plot."""
    download_nltk_packages()
    data = load_json_file(file_path)
    token_lengths = process_poems(data)
    length_percentages = calculate_length_percentages(token_lengths)
    df = create_dataframe(length_percentages)
    create_plot(df)

if __name__ == "__main__":
    main(sys.argv[1])
