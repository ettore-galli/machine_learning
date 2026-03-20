from tokenization.tokenizer import create_word_maps, remove_tokens, tokenize_text
from pytest import mark


@mark.parametrize(
    "text, expected",
    [
        (
            "Sotto la panca, la capra canta.",
            [
                "Sotto",
                " ",
                "la",
                " ",
                "panca",
                ",",
                "",
                " ",
                "la",
                " ",
                "capra",
                " ",
                "canta",
                ".",
                "",
            ],
        )
    ],
)
def test_tokenize_text(text, expected):
    assert tokenize_text(text) == expected


@mark.parametrize(
    "tokenized, to_remove, expected",
    [
        (
            [
                "Sotto",
                " ",
                "la",
                " ",
                "panca",
                ",",
                "",
                " ",
                "la",
                " ",
                "capra",
                " ",
                "canta",
                ".",
                "",
            ],
            [" "],
            [
                "Sotto",
                "la",
                "panca",
                ",",
                "",
                "la",
                "capra",
                "canta",
                ".",
                "",
            ],
        )
    ],
)
def test_remove_tokens(tokenized, to_remove, expected):
    assert remove_tokens(tokenized, to_remove) == expected


def test_create_word_maps():
    assert create_word_maps(
        [
            "Sotto",
            "la",
            "panca",
            ",",
            "",
            "la",
            "capra",
            "canta",
            ".",
            "",
        ]
    ) == (
        {
            "": 0,
            ",": 1,
            ".": 2,
            "Sotto": 3,
            "canta": 4,
            "capra": 5,
            "la": 6,
            "panca": 7,
        },
        {
            0: "",
            1: ",",
            2: ".",
            3: "Sotto",
            4: "canta",
            5: "capra",
            6: "la",
            7: "panca",
        },
    )
