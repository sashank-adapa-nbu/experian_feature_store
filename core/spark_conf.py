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
      - Arrow enabled: faster pandas<->Spark conversions (used in notebooks)
      - Speculation off: avoids re-launching tasks that are legitimately slow
        on 500M-row data

    Call once at notebook startup before running any pipeline method.

    NOTE — spark.speculation is a STATIC config (deploy-time only).
    Spark raises CANNOT_MODIFY_CONFIG if you set it after session creation.
    Set it in the cluster/session Spark config instead:
        Databricks cluster UI -> Advanced Options -> Spark Config:
            spark.speculation false
    This function will warn if the live value differs from the desired value.
    """

    # ── Static configs — cannot be changed after session starts ──────────────
    # We check the live value and warn if it differs from what we want.
    # Fix: set these in the cluster/session config (not here).
    _static_desired = {
        "spark.speculation": str(SPECULATION_ENABLED).lower(),
    }
    for k, desired in _static_desired.items():
        try:
            live = spark.conf.get(k)
        except Exception:
            live = None
        if live != desired:
            logger.warning(
                f"[SparkConf] STATIC config '{k}' is '{live}' (live), "
                f"desired '{desired}'. "
                f"Set it in the cluster/session Spark config before startup."
            )
        else:
            logger.debug(f"  spark.conf (static, already correct): {k} = {live}")

    # ── Runtime-settable configs ──────────────────────────────────────────────
    conf = {
        # ── Shuffle / AQE ─────────────────────────────────────────────────────
        "spark.sql.shuffle.partitions":                           str(SHUFFLE_PARTITIONS),
        "spark.sql.adaptive.enabled":                             str(ADAPTIVE_ENABLED).lower(),
        "spark.sql.adaptive.coalescePartitions.enabled":          "true",
        "spark.sql.adaptive.skewJoin.enabled":                    str(SKEW_JOIN_ENABLED).lower(),
        "spark.sql.adaptive.advisoryPartitionSizeInBytes":        "134217728",  # 128 MB — AQE coalesces to this size
        "spark.sql.adaptive.coalescePartitions.minPartitionNum":  "40",   # min = max_cores (10 workers × 4)

        # ── Join strategy ─────────────────────────────────────────────────────
        "spark.sql.autoBroadcastJoinThreshold":                   str(BROADCAST_THRESHOLD_MB * 1024 * 1024),
        "spark.sql.broadcastTimeout":                             "600",        # 10 min for large masters

        # ── Codegen ───────────────────────────────────────────────────────────
        # grp07/08 produce 250+ column aggs; default 100 triggers fallback to
        # interpreted mode which is 5-10x slower on wide schemas
        "spark.sql.codegen.maxFields":                            str(CODEGEN_MAX_FIELDS),
        "spark.sql.codegen.aggregate.map.twolevel.enabled":       "true",
        "spark.sql.codegen.wholeStage":                           "true",

        # ── Memory ────────────────────────────────────────────────────────────
        "spark.sql.execution.arrow.pyspark.enabled":              str(ARROW_ENABLED).lower(),
        "spark.sql.execution.arrow.maxRecordsPerBatch":           "50000",

        # ── IO / Delta ────────────────────────────────────────────────────────
        "spark.databricks.delta.optimizeWrite.enabled":           "true",
        "spark.databricks.delta.autoCompact.enabled":             "true",
        "spark.sql.files.maxPartitionBytes":                      "134217728",
        "spark.sql.files.openCostInBytes":                        "4194304",

        # ── Datetime ──────────────────────────────────────────────────────────
        # Required for try_to_date on legacy date formats
        "spark.sql.legacy.timeParserPolicy":                      "LEGACY",
    }

    for k, v in conf.items():
        spark.conf.set(k, v)
        logger.debug(f"  spark.conf: {k} = {v}")

    logger.info(
        f"[SparkConf] Applied {len(conf)} runtime settings | "
        f"shuffle_partitions={SHUFFLE_PARTITIONS} | AQE=on | skew=on"
    )
    return spark