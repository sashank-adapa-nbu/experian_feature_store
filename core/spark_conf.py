# core/spark_conf.py
# =============================================================================
# Spark session configuration for 500M+ record scale.
# Called once at pipeline startup before any processing.
# =============================================================================

from pyspark.sql import SparkSession
from config.config import (
    SHUFFLE_PARTITIONS, ADAPTIVE_ENABLED, SKEW_JOIN_ENABLED,
    BROADCAST_THRESHOLD_MB, CODEGEN_MAX_FIELDS, ARROW_ENABLED,
    SPECULATION_ENABLED
)
from core.logger import get_logger

logger = get_logger(__name__)


def configure_spark(spark: SparkSession) -> SparkSession:
    """
    Apply performance-tuned Spark SQL conf for 500M+ row tradeline processing.

    Key settings:
      - AQE on: auto-coalesces post-shuffle partitions, handles skew joins
      - High shuffle partitions: avoid OOM on wide aggregations
      - Codegen field limit raised: grp07/08 produce 250+ column aggregations
      - Broadcast threshold raised: master table joins stay in-memory
      - Arrow enabled: faster pandas↔Spark conversions (used in notebooks)
      - Speculation off: avoids re-launching tasks that are legitimately slow
        on 500M-row data

    Call once at notebook startup before running any pipeline method.
    """
    conf = {
        # ── Shuffle / AQE ─────────────────────────────────────────────────────
        "spark.sql.shuffle.partitions":                       str(SHUFFLE_PARTITIONS),
        "spark.sql.adaptive.enabled":                         str(ADAPTIVE_ENABLED).lower(),
        "spark.sql.adaptive.coalescePartitions.enabled":      "true",
        "spark.sql.adaptive.skewJoin.enabled":                str(SKEW_JOIN_ENABLED).lower(),
        "spark.sql.adaptive.advisoryPartitionSizeInBytes":    "134217728",  # 128 MB
        "spark.sql.adaptive.coalescePartitions.minPartitionNum": "200",

        # ── Join strategy ─────────────────────────────────────────────────────
        "spark.sql.autoBroadcastJoinThreshold":               str(BROADCAST_THRESHOLD_MB * 1024 * 1024),
        "spark.sql.broadcastTimeout":                         "600",        # 10 min for large masters

        # ── Codegen ───────────────────────────────────────────────────────────
        # grp07/08 produce 250+ column aggs; default 100 triggers fallback to
        # interpreted mode which is 5-10x slower on wide schemas
        "spark.sql.codegen.maxFields":                        str(CODEGEN_MAX_FIELDS),
        "spark.sql.codegen.aggregate.map.twolevel.enabled":   "true",
        "spark.sql.codegen.wholeStage":                       "true",

        # ── Memory ────────────────────────────────────────────────────────────
        "spark.sql.execution.arrow.pyspark.enabled":          str(ARROW_ENABLED).lower(),
        "spark.sql.execution.arrow.maxRecordsPerBatch":       "50000",

        # ── IO / Delta ────────────────────────────────────────────────────────
        "spark.databricks.delta.optimizeWrite.enabled":       "true",      # bin-pack small files on write
        "spark.databricks.delta.autoCompact.enabled":         "true",      # compact after write
        "spark.sql.files.maxPartitionBytes":                  "134217728", # 128 MB read chunks
        "spark.sql.files.openCostInBytes":                    "4194304",   # 4 MB open cost

        # ── Speculation ───────────────────────────────────────────────────────
        "spark.speculation":                                  str(SPECULATION_ENABLED).lower(),

        # ── Datetime ──────────────────────────────────────────────────────────
        # Required for try_to_date on legacy date formats
        "spark.sql.legacy.timeParserPolicy":                  "LEGACY",
    }

    for k, v in conf.items():
        spark.conf.set(k, v)
        logger.debug(f"  spark.conf: {k} = {v}")

    logger.info(
        f"[SparkConf] Applied {len(conf)} settings | "
        f"shuffle_partitions={SHUFFLE_PARTITIONS} | AQE=on | skew=on"
    )
    return spark
