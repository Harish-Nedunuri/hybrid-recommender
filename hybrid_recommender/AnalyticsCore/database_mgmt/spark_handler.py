import pyspark.sql
from pyspark.storagelevel import StorageLevel


class SparkMixin:
    """
    Functionality to help handle your spark session.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spark_ = pyspark.sql.SparkSession.builder.getOrCreate()
        self.dbutils_ = None
        self.persisted_ = list()
        self.default_persist_level_ = StorageLevel.MEMORY_AND_DISK

    @property
    def spark(self) -> pyspark.sql.SparkSession:
        assert self.spark_ is not None, "Spark session is not set"
        return self.spark_

    @spark.setter
    def spark(self, spark: pyspark.sql.SparkSession):
        self.spark_ = spark

    @property
    def dbutils(self):
        assert self.spark_ is not None, "Spark session is not set"
        from pyspark.dbutils import DBUtils

        if self.dbutils_ is None:
            self.dbutils_ = DBUtils(self.spark)
        return self.dbutils_

    def set_default_persist_level(self, level: StorageLevel):
        self.default_persist_level_ = level

    def persist(self, df: pyspark.sql.DataFrame, level: StorageLevel = None):
        if level is None:
            level = self.default_persist_level_
        df = df.persist(level)
        self.persisted_.append(df)
        return df

    def __del__(self):
        # Unpersist all
        for df in self.persisted_:
            df.unpersist()
