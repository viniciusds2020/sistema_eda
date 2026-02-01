"""Funções auxiliares do SmartEDA."""

from typing import Union
import numpy as np


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """Formata número para exibição."""
    if pd_isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}".replace(",", ".")
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_percentage(value: float, decimals: int = 2) -> str:
    """Formata percentual para exibição."""
    if pd_isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisão segura que retorna default se denominador é zero."""
    if denominator == 0 or pd_isna(denominator):
        return default
    return numerator / denominator


def truncate_string(s: str, max_length: int = 50) -> str:
    """Trunca string se maior que max_length."""
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def pd_isna(value) -> bool:
    """Verifica se valor é NA/NaN de forma segura."""
    try:
        import pandas as pd
        return pd.isna(value)
    except:
        return value is None or (isinstance(value, float) and np.isnan(value))
