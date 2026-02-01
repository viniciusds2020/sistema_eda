"""Análise de variáveis numéricas."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class NumericStats:
    """Estatísticas de uma variável numérica."""
    column: str
    count: int
    missing: int
    missing_pct: float
    zeros: int
    zeros_pct: float
    mean: float
    std: float
    var: float
    min: float
    max: float
    range: float
    median: float
    mode: float
    skewness: float
    kurtosis: float
    percentiles: Dict[str, float]
    iqr: float
    outliers_iqr: int
    outliers_iqr_pct: float
    outliers_zscore: int
    outliers_zscore_pct: float
    cv: float  # Coeficiente de variação
    normality_pvalue: Optional[float] = None
    is_normal: Optional[bool] = None


class NumericAnalyzer:
    """Analisador de variáveis numéricas."""

    def __init__(
        self,
        percentiles: List[float] = None,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0
    ):
        self.percentiles = percentiles or [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold

    def analyze(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, NumericStats]:
        """Analisa múltiplas colunas numéricas."""
        results = {}
        for col in columns:
            if col in df.columns:
                results[col] = self._analyze_column(df[col])
        return results

    def _analyze_column(self, series: pd.Series) -> NumericStats:
        """Analisa uma coluna numérica."""
        col_name = series.name
        n = len(series)
        missing = series.isna().sum()
        missing_pct = missing / n if n > 0 else 0

        # Remover NaN para cálculos
        data = series.dropna()
        n_valid = len(data)

        if n_valid == 0:
            return self._empty_stats(col_name, n, missing, missing_pct)

        # Estatísticas básicas
        zeros = (data == 0).sum()
        zeros_pct = zeros / n_valid if n_valid > 0 else 0

        mean_val = data.mean()
        std_val = data.std()
        var_val = data.var()
        min_val = data.min()
        max_val = data.max()
        range_val = max_val - min_val
        median_val = data.median()

        # Moda
        mode_result = data.mode()
        mode_val = mode_result.iloc[0] if len(mode_result) > 0 else np.nan

        # Assimetria e curtose
        skewness = stats.skew(data) if n_valid > 2 else np.nan
        kurtosis = stats.kurtosis(data) if n_valid > 3 else np.nan

        # Percentis
        percentile_dict = {}
        for p in self.percentiles:
            pct_key = f"p{int(p * 100)}"
            percentile_dict[pct_key] = data.quantile(p)

        # IQR e outliers
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        outliers_iqr = ((data < lower_bound) | (data > upper_bound)).sum()
        outliers_iqr_pct = outliers_iqr / n_valid if n_valid > 0 else 0

        # Outliers por Z-score
        if std_val > 0:
            z_scores = np.abs((data - mean_val) / std_val)
            outliers_zscore = (z_scores > self.zscore_threshold).sum()
        else:
            outliers_zscore = 0
        outliers_zscore_pct = outliers_zscore / n_valid if n_valid > 0 else 0

        # Coeficiente de variação
        cv = std_val / mean_val if mean_val != 0 else np.nan

        # Teste de normalidade (Shapiro-Wilk, máximo 5000 amostras)
        normality_pvalue = None
        is_normal = None
        if 3 <= n_valid <= 5000:
            try:
                _, normality_pvalue = stats.shapiro(data)
                is_normal = normality_pvalue > 0.05
            except:
                pass
        elif n_valid > 5000:
            # Usar amostra para teste
            sample = data.sample(n=5000, random_state=42)
            try:
                _, normality_pvalue = stats.shapiro(sample)
                is_normal = normality_pvalue > 0.05
            except:
                pass

        return NumericStats(
            column=col_name,
            count=n_valid,
            missing=missing,
            missing_pct=missing_pct,
            zeros=zeros,
            zeros_pct=zeros_pct,
            mean=mean_val,
            std=std_val,
            var=var_val,
            min=min_val,
            max=max_val,
            range=range_val,
            median=median_val,
            mode=mode_val,
            skewness=skewness,
            kurtosis=kurtosis,
            percentiles=percentile_dict,
            iqr=iqr,
            outliers_iqr=outliers_iqr,
            outliers_iqr_pct=outliers_iqr_pct,
            outliers_zscore=outliers_zscore,
            outliers_zscore_pct=outliers_zscore_pct,
            cv=cv,
            normality_pvalue=normality_pvalue,
            is_normal=is_normal
        )

    def _empty_stats(
        self, col_name: str, n: int, missing: int, missing_pct: float
    ) -> NumericStats:
        """Retorna estatísticas vazias para coluna sem dados válidos."""
        return NumericStats(
            column=col_name,
            count=0,
            missing=missing,
            missing_pct=missing_pct,
            zeros=0,
            zeros_pct=0,
            mean=np.nan,
            std=np.nan,
            var=np.nan,
            min=np.nan,
            max=np.nan,
            range=np.nan,
            median=np.nan,
            mode=np.nan,
            skewness=np.nan,
            kurtosis=np.nan,
            percentiles={},
            iqr=np.nan,
            outliers_iqr=0,
            outliers_iqr_pct=0,
            outliers_zscore=0,
            outliers_zscore_pct=0,
            cv=np.nan,
            normality_pvalue=None,
            is_normal=None
        )

    def get_summary_dataframe(self, stats: Dict[str, NumericStats]) -> pd.DataFrame:
        """Converte estatísticas para DataFrame resumo."""
        rows = []
        for col, s in stats.items():
            row = {
                "Variável": s.column,
                "N": s.count,
                "Ausentes": s.missing,
                "Ausentes%": s.missing_pct,
                "Zeros": s.zeros,
                "Média": s.mean,
                "Mediana": s.median,
                "Desvio Padrão": s.std,
                "Mín": s.min,
                "Máx": s.max,
                "Assimetria": s.skewness,
                "Curtose": s.kurtosis,
                "IQR": s.iqr,
                "Outliers IQR": s.outliers_iqr,
                "Normal": "Sim" if s.is_normal else "Não" if s.is_normal is False else "N/A"
            }
            rows.append(row)
        return pd.DataFrame(rows)
