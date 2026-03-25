from pathlib import Path

from fastai.data.block import DataBlock
from fastai.vision.all import (
    ImageBlock,
    CategoryBlock,
    Learner,
    PILImage,
    get_image_files,
    parent_label,
    RandomSplitter,
    Resize,
    vision_learner,
    resnet18,
    error_rate,
)

from fastai_work.base import BIRD_CLASSIFIER_WEIGHTS_FILE, IMAGES_PATH


def get_learner(path: Path) -> Learner:
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method="squish")],
    ).dataloaders(path, bs=32)

    return vision_learner(dls, resnet18, metrics=error_rate)


def perform_model_training(path: Path) -> Learner:
    learn: Learner = get_learner(path=path)
    learn.fine_tune(3)

    learn.save(BIRD_CLASSIFIER_WEIGHTS_FILE)


def perform_prediction(model: Learner, image: Path):
    model.load(BIRD_CLASSIFIER_WEIGHTS_FILE)
    is_bird, _, probs = model.predict(PILImage.create(image))
    print(f"This is a: {is_bird}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")


if __name__ == "__main__":
    perform_model_training(IMAGES_PATH)
    perform_prediction(
        model=get_learner(IMAGES_PATH),
        image=IMAGES_PATH / "undertest" / "1.png",
    )
