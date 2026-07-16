"""Configurações globais do SmartEDA."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configurações para análise exploratória."""

    # Limites para inferência de tipos
    categorical_threshold: int = 20  # Máximo de valores únicos para considerar categórico
    id_unique_ratio: float = 0.95  # Ratio mínimo para considerar como ID

    # Percentis para análise numérica
    percentiles: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    # Outliers
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0

    # Categóricas
    top_n_categories: int = 10
    rare_category_threshold: float = 0.01  # 1%

    # Correlações
    correlation_threshold: float = 0.5  # Correlação significativa

    # Relatório
    include_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 100
    max_categories_plot: int = 15

    # Target
    classification_threshold: int = 10  # Se target tem <= N valores únicos, é classificação

    # Drift e testes estatísticos
    drift_bins: int = 10
    target_bins: int = 5
    min_segment_size: int = 30
    statistical_alpha: float = 0.05
    pvalue_correction: str = "fdr_bh"

    # Geral / materialização limitada
    sample_size: Optional[int] = None  # Limite aplicado antes de coletar Polars/DuckDB
    random_state: int = 42
