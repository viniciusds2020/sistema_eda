"""SmartEDA: statistical profiling and exploratory data analysis."""

from smarteda.analysis.conditioned import target_conditioned_drift
from smarteda.analysis.drift import profile_train_test
from smarteda.analysis.monitoring import monitor_windows
from smarteda.analysis.quality import detect_quality_issues
from smarteda.analysis.statistical_tests import distribution_tests
from smarteda.core.adapters import to_pandas
from smarteda.core.analyzer import SmartEDA
from smarteda.core.config import Config

__version__ = "1.3.0"

__all__ = [
    "SmartEDA",
    "Config",
    "to_pandas",
    "detect_quality_issues",
    "profile_train_test",
    "distribution_tests",
    "target_conditioned_drift",
    "monitor_windows",
]
