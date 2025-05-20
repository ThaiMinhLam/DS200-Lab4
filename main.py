import argparse
from trainer import Trainer
from config import SparkConfig
from models.model import *
from models.extract_features import *

parser = argparse.ArgumentParser(
    description="Streaming Trainer"
)
parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True, 
                    help="Mode to run: train or predict")
parser.add_argument("--model", type=str, choices=["rf", "svm"], required=True, 
                    help="Choosing model: SVM for RF")
parser.add_argument("--save_path", type=str, required=True, 
                    help="Path to save/load model")
args = parser.parse_args()


if __name__ == "__main__":
    spark_config = SparkConfig()
    model = None
    if (args.model == 'rf'):
        model = RandomForest()
    else: 
        model = SVM()
    trainer = Trainer(model, spark_config, args.save_path)

    if (args.mode == 'train'):
        trainer.train()
    else:
        trainer.predict()
