"""Análise de correlações entre variáveis."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CorrelationResult:
    """Resultado de análise de correlação."""
    var1: str
    var2: str
    method: str
    correlation: float
    p_value: Optional[float]
    is_significant: bool


class CorrelationAnalyzer:
    """Analisador de correlações entre variáveis."""

    def __init__(self, significance_level: float = 0.05, min_correlation: float = 0.3):
        self.significance_level = significance_level
        self.min_correlation = min_correlation

    def numeric_correlation(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "pearson"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[CorrelationResult]]:
        """
        Calcula correlação entre variáveis numéricas.

        Returns:
            - Matriz de correlação
            - Matriz de p-valores
            - Lista de correlações significativas
        """
        subset = df[columns].dropna()

        if len(subset) < 3:
            return pd.DataFrame(), pd.DataFrame(), []

        # Matriz de correlação
        if method == "pearson":
            corr_matrix = subset.corr(method="pearson")
        elif method == "spearman":
            corr_matrix = subset.corr(method="spearman")
        else:
            corr_matrix = subset.corr()

        # Matriz de p-valores
        n_cols = len(columns)
        p_matrix = pd.DataFrame(
            np.ones((n_cols, n_cols)),
            index=columns,
            columns=columns
        )

        significant_correlations = []

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:
                    x = subset[col1]
                    y = subset[col2]

                    if method == "pearson":
                        corr, p_value = stats.pearsonr(x, y)
                    elif method == "spearman":
                        corr, p_value = stats.spearmanr(x, y)
                    else:
                        corr, p_value = stats.pearsonr(x, y)

                    p_matrix.loc[col1, col2] = p_value
                    p_matrix.loc[col2, col1] = p_value

                    is_significant = (
                        p_value < self.significance_level and
                        abs(corr) >= self.min_correlation
                    )

                    if is_significant:
                        significant_correlations.append(CorrelationResult(
                            var1=col1,
                            var2=col2,
                            method=method,
                            correlation=corr,
                            p_value=p_value,
                            is_significant=True
                        ))

        # Ordenar por correlação absoluta
        significant_correlations.sort(key=lambda x: abs(x.correlation), reverse=True)

        return corr_matrix, p_matrix, significant_correlations

    def categorical_correlation(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[pd.DataFrame, List[CorrelationResult]]:
        """
        Calcula associação entre variáveis categóricas usando Cramér's V.

        Returns:
            - Matriz de Cramér's V
            - Lista de associações significativas
        """
        n_cols = len(columns)
        cramer_matrix = pd.DataFrame(
            np.eye(n_cols),  # Matriz identidade (diagonal = 1)
            index=columns,
            columns=columns
        )

        significant_associations = []

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:
                    v, p_value = self._cramers_v(df[col1], df[col2])
                    cramer_matrix.loc[col1, col2] = v
                    cramer_matrix.loc[col2, col1] = v

                    is_significant = (
                        p_value is not None and
                        p_value < self.significance_level and
                        v >= self.min_correlation
                    )

                    if is_significant:
                        significant_associations.append(CorrelationResult(
                            var1=col1,
                            var2=col2,
                            method="cramers_v",
                            correlation=v,
                            p_value=p_value,
                            is_significant=True
                        ))

        significant_associations.sort(key=lambda x: abs(x.correlation), reverse=True)

        return cramer_matrix, significant_associations

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> Tuple[float, Optional[float]]:
        """Calcula Cramér's V entre duas variáveis categóricas."""
        # Remover NaN
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return 0.0, None

        # Tabela de contingência
        contingency = pd.crosstab(x_clean, y_clean)

        # Chi-quadrado
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        except:
            return 0.0, None

        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1

        if min_dim == 0 or n == 0:
            return 0.0, p_value

        v = np.sqrt(chi2 / (n * min_dim))
        return v, p_value

    def mixed_correlation(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[CorrelationResult]]:
        """
        Calcula Eta-squared entre variáveis numéricas e categóricas.

        Returns:
            - Matriz de Eta-squared (numéricas nas linhas, categóricas nas colunas)
            - Lista de relações significativas
        """
        eta_matrix = pd.DataFrame(
            index=numeric_cols,
            columns=categorical_cols,
            dtype=float
        )

        significant_relations = []

        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                eta2, f_stat, p_value = self._eta_squared(df, num_col, cat_col)
                eta_matrix.loc[num_col, cat_col] = eta2

                is_significant = (
                    p_value is not None and
                    p_value < self.significance_level and
                    eta2 >= 0.06  # Efeito médio ou maior
                )

                if is_significant:
                    significant_relations.append(CorrelationResult(
                        var1=num_col,
                        var2=cat_col,
                        method="eta_squared",
                        correlation=eta2,
                        p_value=p_value,
                        is_significant=True
                    ))

        significant_relations.sort(key=lambda x: abs(x.correlation), reverse=True)

        return eta_matrix, significant_relations

    def _eta_squared(
        self, df: pd.DataFrame, numeric_col: str, categorical_col: str
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """Calcula Eta-squared e realiza ANOVA."""
        # Limpar dados
        mask = df[numeric_col].notna() & df[categorical_col].notna()
        data = df.loc[mask, [numeric_col, categorical_col]]

        if len(data) < 3:
            return 0.0, None, None

        # Agrupar por categoria
        groups = [group[numeric_col].values for name, group in data.groupby(categorical_col)]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return 0.0, None, None

        # ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*groups)
        except:
            return 0.0, None, None

        # Eta-squared
        grand_mean = data[numeric_col].mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for g in groups
        )
        ss_total = ((data[numeric_col] - grand_mean) ** 2).sum()

        eta2 = ss_between / ss_total if ss_total > 0 else 0

        return eta2, f_stat, p_value

    def point_biserial(
        self, df: pd.DataFrame, numeric_col: str, binary_col: str
    ) -> Tuple[float, float]:
        """Calcula correlação point-biserial entre numérica e binária."""
        mask = df[numeric_col].notna() & df[binary_col].notna()
        x = df.loc[mask, numeric_col]
        y = df.loc[mask, binary_col]

        # Converter binária para 0/1
        unique_vals = y.unique()
        if len(unique_vals) != 2:
            return 0.0, 1.0

        y_binary = (y == unique_vals[0]).astype(int)

        corr, p_value = stats.pointbiserialr(y_binary, x)
        return corr, p_value

    def get_all_significant_correlations(
        self,
        numeric_results: List[CorrelationResult],
        categorical_results: List[CorrelationResult],
        mixed_results: List[CorrelationResult]
    ) -> pd.DataFrame:
        """Consolida todas as correlações significativas em um DataFrame."""
        all_results = numeric_results + categorical_results + mixed_results

        if not all_results:
            return pd.DataFrame()

        rows = []
        for r in all_results:
            rows.append({
                "Variável 1": r.var1,
                "Variável 2": r.var2,
                "Método": r.method,
                "Correlação": r.correlation,
                "P-valor": r.p_value,
                "Significativo": "Sim" if r.is_significant else "Não"
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("Correlação", key=abs, ascending=False)
        return df
