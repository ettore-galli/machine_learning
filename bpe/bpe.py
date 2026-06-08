from collections import defaultdict

import os
from typing import Dict, Generator, Iterable, List, Tuple


def learn_bpe(corpus, num_merges=3):
    vocab = defaultdict(int)

    for sentence in corpus.split("."):
        words = sentence.strip().split()
        for word in words:
            chars = ["<"] + list(word) + [">"]
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                vocab[pair] += 1

    merges = []

    for merge in range(num_merges):
        if not vocab:
            break

        most_frequent = max(vocab, key=lambda x: vocab[x])
        merges.append(most_frequent)

        new_char = "".join(most_frequent)
        new_vocab = defaultdict(int)

        for pair in vocab:
            count = vocab[pair]
            if pair == most_frequent:
                continue
            new_pair = list(pair)
            if new_pair[0] == most_frequent[0] and new_pair[1] == most_frequent[1]:
                new_pair[0] = new_char
                new_pair.pop(1)
            new_vocab[tuple(new_pair)] += count
        vocab = new_vocab

        print(f"\n---------- Merge n° {merge} ----------")
        print(f"vocab={vocab}")
        print(f"merges={merges}")

    return vocab, merges


def apply_bpe(text, merges):
    chars = ["<"] + list(text) + [">"]
    for merge in reversed(merges):
        merged = "".join(merge)
        new_chars = []
        i = 0
        while i < len(chars) - 1:
            if (chars[i], chars[i + 1]) == merge:
                new_chars.append(merged)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        if i < len(chars):
            new_chars.append(chars[-1])
        chars = new_chars

    return chars


def get_filename_full(filename: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def read_corpus(filename_full: str) -> list[str]:
    with open(filename_full) as bc:
        return bc.readlines()


def stream_text_by_words(lines: list[str]) -> Generator[str, None, None]:
    for line in lines:
        for sentence in line.split("."):
            for word in sentence.split():
                yield word


def create_char_pairs(word_chars: list[str]) -> List[Tuple[str, str]]:
    return [(word_chars[i], word_chars[i + 1]) for i in range(len(word_chars) - 1)]


def create_pair_series(
    lines: list[str], end_of_word: str = "_"
) -> Generator[Tuple[str, str], None, None]:
    for word in stream_text_by_words(lines=lines):
        for pair in create_char_pairs(word_chars=list(word + end_of_word)):
            yield pair


def create_base_dict(chars: Iterable[str]) -> Dict[str, int]:
    ngramd = defaultdict(int)
    for char in chars:
        ngramd[char] += 1
    return ngramd


def create_pair_dict(
    pairs: Iterable[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], int]]:
    pairs_echo = []
    encoding = defaultdict(int)
    for pair in pairs:
        pairs_echo.append(pair)
        encoding[pair] += 1

    return pairs_echo, encoding


def encode_step(
    pairs: Iterable[Tuple[str, str]],
    encoding: Dict[Tuple[str, str], int],
    end_of_word: str = "_",
) -> Tuple[Iterable[Tuple[str, str]], Dict[Tuple[str, str], int], bool]:

    pairs_echo, encoding = create_pair_dict(pairs=pairs)

    source_pairs = [item for item in encoding.items() if item[0][1] != end_of_word]

    if not source_pairs:
        return pairs_echo, encoding, False

    top_pair = max(
        [item for item in encoding.items() if item[0][1] != end_of_word],
        key=lambda item: item[1],
    )

    new_pairs = []

    skip = None
    
    for idx, (a, b) in enumerate(pairs_echo):
        if (a, b) == skip:
            continue
        if (a, b) == top_pair[0]:
            new_pairs.append((a + b, pairs_echo[idx + 1][1]))
            skip = pairs_echo[idx + 1]
        else:
            new_pairs.append((a, b))

    return (*create_pair_dict(pairs=new_pairs), True)


def encode_process(source: list[str], steps: int = 10):
    init_pairs = create_pair_series(lines=source)
    pairs, encoding = create_pair_dict(pairs=init_pairs)
    for _ in range(steps):
        pairs, encoding, iterate = encode_step(pairs=pairs, encoding=encoding)
        print("---")
        print(pairs)
        print(encoding)
        if not iterate:
            break
    return pairs, encoding


def main():
    # corpus = """The quick brown fox jumps over the lazy dog"""
    # corpus = """Un uomo è stato ucciso in una sparatoria a nord di Tel Aviv, per la precisione a Kochav Ya'ir, e altre sei persone sono rimaste ferite. La polizia ha catturato un sospetto e dopo una breve caccia all'uomo ha fermato anche un secondo individuo. I media israeliani riferiscono che uno dei terroristi coinvolti nell'attentato, un arabo israeliano, è stato "eliminato". Le forze dello Shin Bet sono state dispiegata nella zona. Nel frattempo l'Idf ha fatto sapere di aver neutralizzato un sospetto operativo di Hamas che partecipò alla strage del 7 ottobre. Intanto, con i continui raid incrociati tra Stati Uniti e Iran e gli attacchi e le tensioni tra Israele ed Hezbollah, soprattutto nel sud del Libano, il cessate il fuoco è sempre più fragile e la pace sembra sempre più lontana, nonostante proseguano i tentativi di negoziato. "Nella giornata di oggi, le forze statunitensi in Medio Oriente hanno abbattuto due droni d'attacco iraniani che minacciavano il traffico marittimo internazionale nello Stretto di Hormuz", si legge in un post su X del Centcom americano."""
    # corpus = read_corpus("bpe/bigcorpus.txt")

    # vocab, merges = learn_bpe(corpus, num_merges=3)
    # print("Vocab:", vocab)
    # print("Learned Merges:", merges)
    # for sentence in corpus.split(".")[:2]:
    #     words = sentence.strip().split()
    #     for word in words[:10]:
    #         bpe_representation = apply_bpe(word, merges)
    #         print(f"BPE Representation for '{word}':", bpe_representation)

    source = ["the cat in the hat thx"]

    # pairs = create_pair_series(lines=source)
    # # pair_dict=create_pair_dict(pairs=pairs)
    # first_step = encode_step(pairs=pairs)
    # print(first_step)

    pairs, encoding = encode_process(source=source)
    print(pairs)
    print(encoding)


if __name__ == "__main__":
    CORPUS_FILE = "bigcorpus.txt"
    main()
