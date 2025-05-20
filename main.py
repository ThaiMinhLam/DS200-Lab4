import argparse
from trainer import Trainer
from config import SparkConfig
from models import SVM

import pyspark
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext


if __name__ == "__main__":

    spark_config = SparkConfig()
    svm = SVM(loss="squared_hinge", penalty="l2")
    trainer = Trainer(svm, "train", spark_config, transforms)
    trainer.train()
    trainer.predict()
