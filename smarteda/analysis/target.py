"""Análise focada em variável target."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TargetAnalysis:
    """Resultado de análise com target."""
    feature: str
    feature_type: str
    target_type: str
    metric_name: str
    metric_value: float
    p_value: Optional[float]
    effect_size: Optional[float]
    interpretation: str


class TargetAnalyzer:
    """Analisador de relações com variável target."""

    def __init__(self, classification_threshold: int = 10):
        self.classification_threshold = classification_threshold

    def analyze(
        self,
        df: pd.DataFrame,
        target: str,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> Dict[str, Any]:
        """Analisa todas as variáveis em relação ao target."""
        target_type = self._determine_target_type(df[target])

        results = {
            "target_name": target,
            "target_type": target_type,
            "target_stats": self._target_stats(df[target], target_type),
            "numeric_analysis": [],
            "categorical_analysis": [],
            "feature_ranking": []
        }

        # Analisar variáveis numéricas
        for col in numeric_cols:
            if col != target:
                analysis = self._analyze_numeric_feature(df, col, target, target_type)
                if analysis:
                    results["numeric_analysis"].append(analysis)

        # Analisar variáveis categóricas
        for col in categorical_cols:
            if col != target:
                analysis = self._analyze_categorical_feature(df, col, target, target_type)
                if analysis:
                    results["categorical_analysis"].append(analysis)

        # Ranking consolidado
        all_analyses = results["numeric_analysis"] + results["categorical_analysis"]
        results["feature_ranking"] = sorted(
            all_analyses,
            key=lambda x: abs(x.metric_value) if x.metric_value is not None else 0,
            reverse=True
        )

        return results

    def _determine_target_type(self, series: pd.Series) -> str:
        """Determina se target é classificação ou regressão."""
        n_unique = series.nunique()

        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= self.classification_threshold:
                return "classification"
            return "regression"
        return "classification"

    def _target_stats(self, series: pd.Series, target_type: str) -> Dict[str, Any]:
        """Estatísticas do target."""
        result = {
            "type": target_type,
            "count": len(series),
            "missing": series.isna().sum(),
            "unique": series.nunique()
        }

        if target_type == "classification":
            value_counts = series.value_counts()
            result["class_distribution"] = value_counts.to_dict()
            result["class_balance"] = value_counts.min() / value_counts.max() if value_counts.max() > 0 else 0
            result["majority_class"] = value_counts.index[0]
            result["majority_pct"] = value_counts.iloc[0] / len(series.dropna())
        else:
            data = series.dropna()
            result["mean"] = data.mean()
            result["std"] = data.std()
            result["min"] = data.min()
            result["max"] = data.max()
            result["median"] = data.median()

        return result

    def _analyze_numeric_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str,
        target_type: str
    ) -> Optional[TargetAnalysis]:
        """Analisa feature numérica vs target."""
        mask = df[feature].notna() & df[target].notna()
        data = df.loc[mask, [feature, target]]

        if len(data) < 10:
            return None

        if target_type == "classification":
            return self._numeric_vs_classification(data, feature, target)
        else:
            return self._numeric_vs_regression(data, feature, target)

    def _numeric_vs_classification(
        self, data: pd.DataFrame, feature: str, target: str
    ) -> TargetAnalysis:
        """Analisa numérica vs target de classificação (ANOVA)."""
        groups = [group[feature].values for name, group in data.groupby(target)]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            return TargetAnalysis(
                feature=feature,
                feature_type="numeric",
                target_type="classification",
                metric_name="F-statistic",
                metric_value=0,
                p_value=1.0,
                effect_size=0,
                interpretation="Dados insuficientes"
            )

        try:
            f_stat, p_value = stats.f_oneway(*groups)
        except:
            f_stat, p_value = 0, 1.0

        # Eta-squared
        grand_mean = data[feature].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = ((data[feature] - grand_mean) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else 0

        # Interpretação
        if p_value > 0.05:
            interpretation = "Sem diferença significativa entre classes"
        elif eta2 < 0.06:
            interpretation = "Diferença pequena entre classes"
        elif eta2 < 0.14:
            interpretation = "Diferença média entre classes"
        else:
            interpretation = "Diferença grande entre classes"

        return TargetAnalysis(
            feature=feature,
            feature_type="numeric",
            target_type="classification",
            metric_name="Eta-squared",
            metric_value=eta2,
            p_value=p_value,
            effect_size=eta2,
            interpretation=interpretation
        )

    def _numeric_vs_regression(
        self, data: pd.DataFrame, feature: str, target: str
    ) -> TargetAnalysis:
        """Analisa numérica vs target de regressão (correlação)."""
        corr, p_value = stats.pearsonr(data[feature], data[target])

        # Interpretação
        abs_corr = abs(corr)
        if p_value > 0.05:
            interpretation = "Correlação não significativa"
        elif abs_corr < 0.3:
            interpretation = "Correlação fraca"
        elif abs_corr < 0.5:
            interpretation = "Correlação moderada"
        elif abs_corr < 0.7:
            interpretation = "Correlação forte"
        else:
            interpretation = "Correlação muito forte"

        direction = "positiva" if corr > 0 else "negativa"
        interpretation = f"{interpretation} ({direction})"

        return TargetAnalysis(
            feature=feature,
            feature_type="numeric",
            target_type="regression",
            metric_name="Correlação Pearson",
            metric_value=corr,
            p_value=p_value,
            effect_size=corr ** 2,  # R²
            interpretation=interpretation
        )

    def _analyze_categorical_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str,
        target_type: str
    ) -> Optional[TargetAnalysis]:
        """Analisa feature categórica vs target."""
        mask = df[feature].notna() & df[target].notna()
        data = df.loc[mask, [feature, target]]

        if len(data) < 10:
            return None

        if target_type == "classification":
            return self._categorical_vs_classification(data, feature, target)
        else:
            return self._categorical_vs_regression(data, feature, target)

    def _categorical_vs_classification(
        self, data: pd.DataFrame, feature: str, target: str
    ) -> TargetAnalysis:
        """Analisa categórica vs target de classificação (Chi-quadrado + Cramér's V)."""
        contingency = pd.crosstab(data[feature], data[target])

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        except:
            chi2, p_value = 0, 1.0

        # Cramér's V
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0

        # Interpretação
        if p_value > 0.05:
            interpretation = "Associação não significativa"
        elif v < 0.1:
            interpretation = "Associação negligenciável"
        elif v < 0.3:
            interpretation = "Associação fraca"
        elif v < 0.5:
            interpretation = "Associação moderada"
        else:
            interpretation = "Associação forte"

        return TargetAnalysis(
            feature=feature,
            feature_type="categorical",
            target_type="classification",
            metric_name="Cramér's V",
            metric_value=v,
            p_value=p_value,
            effect_size=v,
            interpretation=interpretation
        )

    def _categorical_vs_regression(
        self, data: pd.DataFrame, feature: str, target: str
    ) -> TargetAnalysis:
        """Analisa categórica vs target de regressão (ANOVA)."""
        groups = [group[target].values for name, group in data.groupby(feature)]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            return TargetAnalysis(
                feature=feature,
                feature_type="categorical",
                target_type="regression",
                metric_name="Eta-squared",
                metric_value=0,
                p_value=1.0,
                effect_size=0,
                interpretation="Dados insuficientes"
            )

        try:
            f_stat, p_value = stats.f_oneway(*groups)
        except:
            f_stat, p_value = 0, 1.0

        # Eta-squared
        grand_mean = data[target].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = ((data[target] - grand_mean) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else 0

        # Interpretação
        if p_value > 0.05:
            interpretation = "Sem diferença significativa entre grupos"
        elif eta2 < 0.06:
            interpretation = "Efeito pequeno"
        elif eta2 < 0.14:
            interpretation = "Efeito médio"
        else:
            interpretation = "Efeito grande"

        return TargetAnalysis(
            feature=feature,
            feature_type="categorical",
            target_type="regression",
            metric_name="Eta-squared",
            metric_value=eta2,
            p_value=p_value,
            effect_size=eta2,
            interpretation=interpretation
        )

    def information_value(
        self, df: pd.DataFrame, feature: str, target: str, bins: int = 10
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calcula Information Value (IV) para classificação binária.

        IV < 0.02: Não preditivo
        0.02 - 0.1: Fraco
        0.1 - 0.3: Médio
        0.3 - 0.5: Forte
        > 0.5: Muito forte (suspeito)
        """
        mask = df[feature].notna() & df[target].notna()
        data = df.loc[mask, [feature, target]].copy()

        # Target deve ser binário
        unique_target = data[target].unique()
        if len(unique_target) != 2:
            return 0.0, pd.DataFrame()

        # Converter target para 0/1
        data["target_binary"] = (data[target] == unique_target[0]).astype(int)

        # Se feature é numérica, criar bins
        if pd.api.types.is_numeric_dtype(data[feature]):
            data["feature_binned"] = pd.qcut(data[feature], bins, duplicates="drop")
        else:
            data["feature_binned"] = data[feature]

        # Calcular WoE e IV
        total_events = data["target_binary"].sum()
        total_non_events = len(data) - total_events

        iv_table = data.groupby("feature_binned").agg(
            count=("target_binary", "count"),
            events=("target_binary", "sum")
        ).reset_index()

        iv_table["non_events"] = iv_table["count"] - iv_table["events"]
        iv_table["pct_events"] = iv_table["events"] / total_events
        iv_table["pct_non_events"] = iv_table["non_events"] / total_non_events

        # Evitar divisão por zero
        iv_table["pct_events"] = iv_table["pct_events"].replace(0, 0.0001)
        iv_table["pct_non_events"] = iv_table["pct_non_events"].replace(0, 0.0001)

        iv_table["woe"] = np.log(iv_table["pct_non_events"] / iv_table["pct_events"])
        iv_table["iv"] = (iv_table["pct_non_events"] - iv_table["pct_events"]) * iv_table["woe"]

        total_iv = iv_table["iv"].sum()

        return total_iv, iv_table

    def get_summary_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Converte resultados para DataFrame resumo."""
        ranking = results.get("feature_ranking", [])

        if not ranking:
            return pd.DataFrame()

        rows = []
        for r in ranking:
            rows.append({
                "Feature": r.feature,
                "Tipo Feature": r.feature_type,
                "Métrica": r.metric_name,
                "Valor": r.metric_value,
                "P-valor": r.p_value,
                "Tamanho Efeito": r.effect_size,
                "Interpretação": r.interpretation
            })

        return pd.DataFrame(rows)
