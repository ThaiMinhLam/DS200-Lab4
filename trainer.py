import pyspark
from pyspark import RDD
from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataloader import StreamingDataLoader
from config import SparkConfig

class Trainer:
    def __init__(self, 
        model, 
        spark_config: SparkConfig, 
        save_path: str
    ) -> None:

        self.model = model
        self.sparkConf = spark_config
        self.save_path = save_path

        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]", self.sparkConf.appName)
        self.ssc = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = StreamingDataLoader(self.sc, self.ssc, self.sqlContext, self.sparkConf)
        
        self.total_batches = 0

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)
        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: RDD) -> DataFrame:
        if not rdd.isEmpty():
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)
            ])

            df = self.sqlContext.createDataFrame(rdd, schema)

            predictions, accuracy, precision, recall, f1 = self.model.train(df)

            print("=" * 10)
            print(f"Predictions: {predictions}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("=" * 10)
            
            self.model.save(self.save_path)
            print(f"Model saved to {self.save_path}")
        else:
            print("No data received. Skipping training.")
            return
        
        print("Total Batch Size of RDD Received:", rdd.count())
        print("+" * 20)

    def predict(self):
        print(f"Loading model from {self.save_path} ...")
        self.model.load(self.save_path)
        self.empty_batches = 0
        self.y_preds = []
        self.y_trues = []
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__predict__)
        self.ssc.start()
        self.ssc.awaitTermination()

    def __predict__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:     
        if not rdd.isEmpty():
            schema = StructType([
                StructField(name="image", dataType=VectorUDT(), nullable=True),
                StructField(name="label",dataType=IntegerType(),nullable=True)])
            
            df = self.sqlContext.createDataFrame(rdd, schema)
            
            trues, preds, acc, prec, rec, f1 = self.model.predict(df)
            self.y_preds.extend(preds.tolist())
            self.y_trues.extend(trues.tolist())
            self.total_batches += 1

            print(f"Test Accuracy : ", acc)
            print(f"Test Precision :", prec)
            print(f"Test Recall : ", rec)
            print(f"Test F1 Score: ", f1)
            
            print("+" * 20)
        else:
            self.empty_batches += 1
            if self.empty_batches >= 3:
                print("="*20)
                print(f"Total Batch Size of RDD Received: {self.total_batches}")
                print(f"Final Accuracy : {accuracy_score(self.y_trues, self.y_preds):.4f}")
                print(f"Final Precision: {precision_score(self.y_trues, self.y_preds, average='macro'):.4f}")
                print(f"Final Recall   : {recall_score(self.y_trues, self.y_preds, average='macro'):.4f}")
                print(f"Final F1-score : {f1_score(self.y_trues, self.y_preds, average='macro'):.4f}")
                print("="*20)
                self.ssc.stop(stopSparkContext=True, stopGraceFully=True)