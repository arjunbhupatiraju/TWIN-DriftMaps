"""driftaware: a drift-aware test for neural encoding of slowly varying task variables."""
from .core import (DriftTestResult, ar_bic, drift_stat, drift_test, imposter_labels, plain_stat,
                   remove_shared_drift, shuffle_labels)
from .maps import bh, fisher, region_map
from .simulate import simulate_session

__version__ = "0.1.0"
__all__ = ["drift_test", "DriftTestResult", "ar_bic", "drift_stat", "plain_stat", "remove_shared_drift",
           "imposter_labels", "shuffle_labels", "fisher", "bh", "region_map", "simulate_session"]
