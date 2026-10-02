from llmgpt.config import get_config
from llmgpt.dataset import X, y
from llmgpt.model import model
from llmgpt.train import train_model

if __name__ == "__main__":
    train_model(config=get_config(), model=model, X=X, y=y)
