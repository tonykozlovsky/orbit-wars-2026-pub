"""IMPALA training configuration — re-exports from :mod:`configs.base`."""

from .base import (
    FrozenOpponentConfig,
    ImpalaTrainingConfig,
    LoggingConfig,
    OptimizerConfig,
    TeacherConfig,
    deep_merge_dicts,
    default_training_config_dict,
)

__all__ = [
    "FrozenOpponentConfig",
    "ImpalaTrainingConfig",
    "LoggingConfig",
    "OptimizerConfig",
    "TeacherConfig",
    "deep_merge_dicts",
    "default_training_config_dict",
]
