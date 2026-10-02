from common import get_config
from dataset import X, y
from model import model
from train import train_model

if __name__ == "__main__":
    train_model(config=get_config(), model=model, X=X, y=y)
