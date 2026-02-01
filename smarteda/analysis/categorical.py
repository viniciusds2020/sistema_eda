"""Análise de variáveis categóricas."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CategoryStats:
    """Estatísticas de uma variável categórica."""
    column: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    cardinality_ratio: float
    mode: str
    mode_count: int
    mode_pct: float
    entropy: float
    top_categories: List[Dict[str, Any]]
    rare_categories: List[str]
    rare_count: int


class CategoricalAnalyzer:
    """Analisador de variáveis categóricas."""

    def __init__(self, top_n: int = 10, rare_threshold: float = 0.01):
        self.top_n = top_n
        self.rare_threshold = rare_threshold

    def analyze(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, CategoryStats]:
        """Analisa múltiplas colunas categóricas."""
        results = {}
        for col in columns:
            if col in df.columns:
                results[col] = self._analyze_column(df[col])
        return results

    def _analyze_column(self, series: pd.Series) -> CategoryStats:
        """Analisa uma coluna categórica."""
        col_name = series.name
        n = len(series)
        missing = series.isna().sum()
        missing_pct = missing / n if n > 0 else 0

        # Dados não nulos
        data = series.dropna()
        n_valid = len(data)

        if n_valid == 0:
            return self._empty_stats(col_name, n, missing, missing_pct)

        # Contagens
        value_counts = data.value_counts()
        unique = len(value_counts)
        cardinality_ratio = unique / n_valid if n_valid > 0 else 0

        # Moda
        mode_val = value_counts.index[0] if len(value_counts) > 0 else None
        mode_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
        mode_pct = mode_count / n_valid if n_valid > 0 else 0

        # Entropia (Shannon)
        probabilities = value_counts / n_valid
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Top N categorias
        top_categories = []
        for i, (cat, count) in enumerate(value_counts.head(self.top_n).items()):
            top_categories.append({
                "rank": i + 1,
                "category": str(cat),
                "count": count,
                "percentage": count / n_valid
            })

        # Categorias raras
        rare_mask = (value_counts / n_valid) < self.rare_threshold
        rare_categories = value_counts[rare_mask].index.tolist()
        rare_count = len(rare_categories)

        return CategoryStats(
            column=col_name,
            count=n_valid,
            missing=missing,
            missing_pct=missing_pct,
            unique=unique,
            cardinality_ratio=cardinality_ratio,
            mode=str(mode_val) if mode_val is not None else "N/A",
            mode_count=mode_count,
            mode_pct=mode_pct,
            entropy=entropy,
            top_categories=top_categories,
            rare_categories=[str(c) for c in rare_categories[:10]],  # Limitar lista
            rare_count=rare_count
        )

    def _empty_stats(
        self, col_name: str, n: int, missing: int, missing_pct: float
    ) -> CategoryStats:
        """Retorna estatísticas vazias."""
        return CategoryStats(
            column=col_name,
            count=0,
            missing=missing,
            missing_pct=missing_pct,
            unique=0,
            cardinality_ratio=0,
            mode="N/A",
            mode_count=0,
            mode_pct=0,
            entropy=0,
            top_categories=[],
            rare_categories=[],
            rare_count=0
        )

    def get_summary_dataframe(self, stats: Dict[str, CategoryStats]) -> pd.DataFrame:
        """Converte estatísticas para DataFrame resumo."""
        rows = []
        for col, s in stats.items():
            row = {
                "Variável": s.column,
                "N": s.count,
                "Ausentes": s.missing,
                "Ausentes%": s.missing_pct,
                "Únicos": s.unique,
                "Cardinalidade": s.cardinality_ratio,
                "Moda": s.mode,
                "Moda%": s.mode_pct,
                "Entropia": s.entropy,
                "Raros": s.rare_count
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def get_frequency_table(self, series: pd.Series) -> pd.DataFrame:
        """Gera tabela de frequência para uma variável."""
        value_counts = series.value_counts()
        n = len(series.dropna())

        df = pd.DataFrame({
            "Categoria": value_counts.index,
            "Frequência": value_counts.values,
            "Percentual": value_counts.values / n if n > 0 else 0,
            "Acumulado": np.cumsum(value_counts.values) / n if n > 0 else 0
        })
        return df
