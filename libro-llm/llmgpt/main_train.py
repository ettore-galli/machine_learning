from llmgpt.config import get_config
from llmgpt.dataset import X, y
from llmgpt.model import GptNN
from llmgpt.train import train_model

if __name__ == "__main__":
    model = GptNN(config=get_config())
    train_model(config=get_config(), model=model, X=X, y=y)
