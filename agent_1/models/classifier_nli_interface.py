# pyright: reportUnknownMemberType=none
# pyright: reportUnknownArgumentType=none
# pyright: reportPrivateImportUsage=none

from typing import Dict, Tuple, cast

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from models.classifier_model_base import KeywordArgsType


def prepare_input_tensor_for_device(
    device: str, input_tensor: torch.Tensor
) -> torch.Tensor:
    return (
        input_tensor.to(device=device).contiguous() if device != "cpu" else input_tensor
    )


def instantiate_classifier_objects(
    model_id: str, device: str
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_id))

    model: PreTrainedModel = cast(
        PreTrainedModel,
        AutoModelForSequenceClassification.from_pretrained(
            model_id,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ),
    )

    model.to(device)

    if device == "mps":
        for param in model.parameters():
            param.data = param.data.to(device)

    return tokenizer, model


def do_classifier_perform(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    text: str,
    text_pair: str,
    **kwargs: KeywordArgsType,
) -> Dict[str, float]:

    encoded: BatchEncoding = tokenizer(
        text,
        text_pair,
        return_tensors="pt",
    )

    inputs: torch.Tensor = prepare_input_tensor_for_device(
        device=device, input_tensor=encoded["input_ids"]
    )

    with torch.no_grad():
        outputs = model(inputs, **kwargs)
        probs = outputs.logits.softmax(dim=1)

    id2label: dict[int, str] = cast(dict[int, str], model.config.id2label)

    return {label: probs[0][result_id].item() for result_id, label in id2label.items()}
