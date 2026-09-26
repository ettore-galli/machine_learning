from dataset import X, y
from model import model
from train import train_model
from common import get_config

if __name__ == "__main__":
    train_model(config=get_config(), model=model, X=X, y=y)
