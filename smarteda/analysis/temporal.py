"""Análise de variáveis temporais."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TemporalStats:
    """Estatísticas de uma variável temporal."""
    column: str
    count: int
    missing: int
    missing_pct: float
    min_date: pd.Timestamp
    max_date: pd.Timestamp
    range_days: int
    range_str: str
    has_time: bool
    weekday_distribution: Dict[str, int]
    month_distribution: Dict[str, int]
    year_distribution: Dict[int, int]
    gaps: List[Dict[str, Any]]
    most_common_weekday: str
    most_common_month: str


class TemporalAnalyzer:
    """Analisador de variáveis temporais."""

    WEEKDAYS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    MONTHS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    def analyze(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, TemporalStats]:
        """Analisa múltiplas colunas temporais."""
        results = {}
        for col in columns:
            if col in df.columns:
                results[col] = self._analyze_column(df[col])
        return results

    def _analyze_column(self, series: pd.Series) -> TemporalStats:
        """Analisa uma coluna temporal."""
        col_name = series.name
        n = len(series)
        missing = series.isna().sum()
        missing_pct = missing / n if n > 0 else 0

        # Converter para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                series = pd.to_datetime(series, errors='coerce')
            except:
                return self._empty_stats(col_name, n, missing, missing_pct)

        data = series.dropna()
        n_valid = len(data)

        if n_valid == 0:
            return self._empty_stats(col_name, n, missing, missing_pct)

        # Range
        min_date = data.min()
        max_date = data.max()
        range_days = (max_date - min_date).days

        # Formatar range
        if range_days > 365:
            years = range_days / 365
            range_str = f"{years:.1f} anos"
        elif range_days > 30:
            months = range_days / 30
            range_str = f"{months:.1f} meses"
        else:
            range_str = f"{range_days} dias"

        # Verificar se tem componente de tempo
        has_time = (data.dt.hour != 0).any() or (data.dt.minute != 0).any()

        # Distribuição por dia da semana
        weekday_counts = data.dt.dayofweek.value_counts().sort_index()
        weekday_distribution = {
            self.WEEKDAYS[i]: int(weekday_counts.get(i, 0))
            for i in range(7)
        }
        most_common_weekday = self.WEEKDAYS[weekday_counts.idxmax()] if len(weekday_counts) > 0 else "N/A"

        # Distribuição por mês
        month_counts = data.dt.month.value_counts().sort_index()
        month_distribution = {
            self.MONTHS[i-1]: int(month_counts.get(i, 0))
            for i in range(1, 13)
        }
        most_common_month = self.MONTHS[month_counts.idxmax() - 1] if len(month_counts) > 0 else "N/A"

        # Distribuição por ano
        year_counts = data.dt.year.value_counts().sort_index()
        year_distribution = {int(k): int(v) for k, v in year_counts.items()}

        # Detectar gaps significativos (mais de 7 dias entre registros consecutivos)
        gaps = self._detect_gaps(data, threshold_days=7)

        return TemporalStats(
            column=col_name,
            count=n_valid,
            missing=missing,
            missing_pct=missing_pct,
            min_date=min_date,
            max_date=max_date,
            range_days=range_days,
            range_str=range_str,
            has_time=has_time,
            weekday_distribution=weekday_distribution,
            month_distribution=month_distribution,
            year_distribution=year_distribution,
            gaps=gaps,
            most_common_weekday=most_common_weekday,
            most_common_month=most_common_month
        )

    def _detect_gaps(
        self, series: pd.Series, threshold_days: int = 7, max_gaps: int = 5
    ) -> List[Dict[str, Any]]:
        """Detecta gaps significativos na série temporal."""
        sorted_dates = series.sort_values()
        diffs = sorted_dates.diff().dropna()

        # Converter para dias
        diffs_days = diffs.dt.days

        # Encontrar gaps maiores que threshold
        large_gaps = diffs_days[diffs_days > threshold_days]

        gaps = []
        for idx in large_gaps.head(max_gaps).index:
            pos = sorted_dates.index.get_loc(idx)
            if pos > 0:
                prev_idx = sorted_dates.index[pos - 1]
                gaps.append({
                    "start": sorted_dates[prev_idx],
                    "end": sorted_dates[idx],
                    "days": int(diffs_days[idx])
                })

        return gaps

    def _empty_stats(
        self, col_name: str, n: int, missing: int, missing_pct: float
    ) -> TemporalStats:
        """Retorna estatísticas vazias."""
        return TemporalStats(
            column=col_name,
            count=0,
            missing=missing,
            missing_pct=missing_pct,
            min_date=pd.NaT,
            max_date=pd.NaT,
            range_days=0,
            range_str="N/A",
            has_time=False,
            weekday_distribution={},
            month_distribution={},
            year_distribution={},
            gaps=[],
            most_common_weekday="N/A",
            most_common_month="N/A"
        )

    def get_summary_dataframe(self, stats: Dict[str, TemporalStats]) -> pd.DataFrame:
        """Converte estatísticas para DataFrame resumo."""
        rows = []
        for col, s in stats.items():
            row = {
                "Variável": s.column,
                "N": s.count,
                "Ausentes": s.missing,
                "Mín": s.min_date,
                "Máx": s.max_date,
                "Range": s.range_str,
                "Tem Hora": "Sim" if s.has_time else "Não",
                "Dia Mais Comum": s.most_common_weekday,
                "Mês Mais Comum": s.most_common_month,
                "Gaps Detectados": len(s.gaps)
            }
            rows.append(row)
        return pd.DataFrame(rows)
