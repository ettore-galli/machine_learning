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

from models.classifier_model_base import KeywordArgsType, ModelClassifierNLIProtocol


def infer_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return "cpu"


def prepare_input_tensor_for_device(
    device: str, input_tensor: torch.Tensor
) -> torch.Tensor:
    return (
        input_tensor.to(device=device).contiguous() if device != "cpu" else input_tensor
    )


def move_to_device(model: PreTrainedModel, device: str) -> PreTrainedModel:
    model.to(device)  # pyright: ignore[reportArgumentType]
    return model


def instantiate_classifier_objects(
    model_id: str, device: str
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:

    tokenizer: PreTrainedTokenizer
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_id))

    model: PreTrainedModel
    model = cast(
        PreTrainedModel,
        AutoModelForSequenceClassification.from_pretrained(
            model_id,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ),
    )

    model = move_to_device(model, device)

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


def get_model_performer(model_id: str) -> ModelClassifierNLIProtocol:

    device = infer_device()
    tokenizer, model = instantiate_classifier_objects(model_id=model_id, device=device)

    def model_performer(
        text: str, text_pair: str, **kwargs: KeywordArgsType
    ) -> Dict[str, float]:

        result = do_classifier_perform(
            tokenizer=tokenizer,
            model=model,
            device=device,
            text=text,
            text_pair=text_pair,
            **kwargs,
        )

        return result

    return model_performer
