"""SmartEDA: statistical profiling and exploratory data analysis."""

from smarteda.analysis.drift import profile_train_test
from smarteda.analysis.quality import detect_quality_issues
from smarteda.core.adapters import to_pandas
from smarteda.core.analyzer import SmartEDA
from smarteda.core.config import Config

__version__ = "1.2.0"

__all__ = [
    "SmartEDA",
    "Config",
    "to_pandas",
    "detect_quality_issues",
    "profile_train_test",
]
