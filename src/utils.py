import json
from typing import Dict, List, Union

def load_json(path: str) -> Union[Dict, List]:
    """
    Load JSON file from a given path.
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: str, data: Union[Dict, List]) -> None:
    """
    Save data to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
