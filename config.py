class SparkConfig:
    appName = "MedicalImage"
    receivers = 2
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 5