import re
from typing import Dict, List, Tuple


def tokenize_text(source: str) -> List[str]:
    return re.split(r'([,.;:?_\-!"()\']|--|\s)', source)


def remove_tokens(tokenized: List[str], tokens_to_remove: List[str]) -> List[str]:
    return [token for token in tokenized if token not in tokens_to_remove]


def create_word_maps(tokenized: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    word_to_id: Dict[str, int] = {item: idx for idx, item in enumerate(sorted(set(tokenized)))}
    id_to_word: Dict[int, str] = {value: key for key, value in word_to_id.items()}
    return word_to_id, id_to_word

 