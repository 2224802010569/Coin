from dataclasses import fields
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType,
)


class SparkService:
    _spark: SparkSession = None

    @classmethod
    def spark(cls) -> SparkSession:
        if cls._spark is None:
            cls._spark = (
                SparkSession.builder
                .appName("candle")
                .getOrCreate()
            )
        return cls._spark

    @staticmethod
    def schema_from_entity(entity_cls) -> StructType:
        python_to_spark = {
            int: IntegerType(),
            float: DoubleType(),
            str: StringType(),
        }
        spark_fields = []
        for f in fields(entity_cls):
            if f.type is datetime:
                spark_type = TimestampType()
            else:
                spark_type = python_to_spark.get(f.type, StringType())
            spark_fields.append(
                StructField(f.name, spark_type, nullable=True)
            )
        return StructType(spark_fields)
