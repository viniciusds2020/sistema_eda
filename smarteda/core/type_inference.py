"""Inferência automática de tipos de dados."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    """Tipos de dados detectados."""
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Informações sobre uma coluna."""
    name: str
    original_dtype: str
    inferred_type: DataType
    unique_count: int
    null_count: int
    null_percentage: float
    sample_values: List
    recommended_dtype: Optional[str] = None


class TypeInference:
    """Classe para inferência automática de tipos de dados."""

    def __init__(self, categorical_threshold: int = 20, id_unique_ratio: float = 0.95):
        self.categorical_threshold = categorical_threshold
        self.id_unique_ratio = id_unique_ratio

    def infer_types(self, df: pd.DataFrame) -> Dict[str, ColumnInfo]:
        """Infere tipos de todas as colunas do DataFrame."""
        result = {}
        for col in df.columns:
            result[col] = self._analyze_column(df, col)
        return result

    def _analyze_column(self, df: pd.DataFrame, col: str) -> ColumnInfo:
        """Analisa uma coluna específica."""
        series = df[col]
        n_rows = len(df)
        unique_count = series.nunique()
        null_count = series.isna().sum()
        null_percentage = null_count / n_rows if n_rows > 0 else 0

        # Sample values (non-null)
        non_null = series.dropna()
        sample_values = non_null.head(5).tolist() if len(non_null) > 0 else []

        # Inferir tipo
        inferred_type, recommended_dtype = self._infer_single_type(
            series, unique_count, n_rows
        )

        return ColumnInfo(
            name=col,
            original_dtype=str(series.dtype),
            inferred_type=inferred_type,
            unique_count=unique_count,
            null_count=null_count,
            null_percentage=null_percentage,
            sample_values=sample_values,
            recommended_dtype=recommended_dtype
        )

    def _infer_single_type(
        self, series: pd.Series, unique_count: int, n_rows: int
    ) -> Tuple[DataType, Optional[str]]:
        """Infere o tipo de uma série."""

        # Constante
        if unique_count <= 1:
            return DataType.CONSTANT, None

        # Já é datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME, "datetime64[ns]"

        # Tentar converter para datetime
        if series.dtype == object:
            try:
                parsed = pd.to_datetime(series.dropna().head(100), errors='coerce')
                if parsed.notna().sum() > len(parsed) * 0.8:
                    return DataType.DATETIME, "datetime64[ns]"
            except:
                pass

        # Binário
        if unique_count == 2:
            return DataType.BINARY, "category"

        # Numérico
        if pd.api.types.is_numeric_dtype(series):
            # Verificar se é ID
            unique_ratio = unique_count / n_rows if n_rows > 0 else 0
            if unique_ratio >= self.id_unique_ratio:
                return DataType.ID, None

            # Discreto vs Contínuo
            if pd.api.types.is_integer_dtype(series):
                if unique_count <= self.categorical_threshold:
                    return DataType.NUMERIC_DISCRETE, "int64"
                return DataType.NUMERIC_DISCRETE, "int64"
            else:
                return DataType.NUMERIC_CONTINUOUS, "float64"

        # String/Object
        if series.dtype == object:
            # Verificar se é ID
            unique_ratio = unique_count / n_rows if n_rows > 0 else 0
            if unique_ratio >= self.id_unique_ratio:
                return DataType.ID, None

            # Verificar se parece numérico
            try:
                numeric = pd.to_numeric(series.dropna().head(100), errors='coerce')
                if numeric.notna().sum() > len(numeric) * 0.8:
                    return DataType.NUMERIC_CONTINUOUS, "float64"
            except:
                pass

            # Categórico vs Texto
            if unique_count <= self.categorical_threshold:
                return DataType.CATEGORICAL, "category"

            # Verificar comprimento médio do texto
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 50:
                return DataType.TEXT, None
            else:
                return DataType.CATEGORICAL, "category"

        return DataType.UNKNOWN, None

    def apply_conversions(self, df: pd.DataFrame, column_info: Dict[str, ColumnInfo]) -> pd.DataFrame:
        """Aplica conversões de tipo recomendadas."""
        df_converted = df.copy()

        for col, info in column_info.items():
            if info.recommended_dtype is None:
                continue

            try:
                if info.recommended_dtype == "datetime64[ns]":
                    df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                elif info.recommended_dtype == "category":
                    df_converted[col] = df_converted[col].astype("category")
                elif info.recommended_dtype == "float64":
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                elif info.recommended_dtype == "int64":
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            except Exception:
                pass  # Manter tipo original se conversão falhar

        return df_converted

    def get_columns_by_type(
        self, column_info: Dict[str, ColumnInfo], data_types: List[DataType]
    ) -> List[str]:
        """Retorna lista de colunas de determinados tipos."""
        return [
            col for col, info in column_info.items()
            if info.inferred_type in data_types
        ]
