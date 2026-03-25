from fastai_work.base import IMAGES_PATH
from fastai_work.model_core import get_learner, perform_prediction


if __name__ == "__main__":
    perform_prediction(
        model=get_learner(IMAGES_PATH),
        image=IMAGES_PATH / "undertest" / "1.jpeg",
    )
