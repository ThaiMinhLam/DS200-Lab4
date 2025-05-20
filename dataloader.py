import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector

from config import SparkConfig

import json

class DataLoader:
    def __init__(self, 
        sparkContext:SparkContext, 
        sparkStreamingContext: StreamingContext, 
        sqlContext: SQLContext,
        sparkConf: SparkConfig
    ) -> None:
        
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port
        )


    def parse_stream(self) -> DStream:
        record_stream = (
            self.stream
            .map(lambda line: json.loads(line))                           # Parse JSON string to dict
            .flatMap(lambda record_dict: record_dict.values())            # Flatten batch records
            .map(lambda record: list(record.values()))                    # Convert dict to list
            .map(lambda values: [
                np.array(values[:-1], dtype=np.uint8)                     # All except last are features
                .reshape(3, 32, 32)                                       # Shape: (3, 32, 32)
                .transpose(1, 2, 0),                                      # Convert to (32, 32, 3)
                int(values[-1])                                           # Last element is the label
            ])
        )
        
        return self.preprocess(record_stream)

    @staticmethod
    def preprocess(stream: DStream) -> DStream:
        stream = stream.map(lambda x: [x[0].reshape(-1).tolist(),x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]), x[1]])
        
        return stream
