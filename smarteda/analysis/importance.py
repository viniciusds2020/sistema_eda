"""Análise de importância de variáveis."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.feature_selection import f_classif, f_regression
    from sklearn.feature_selection import chi2
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class FeatureImportance:
    """Importância de uma feature."""
    feature: str
    feature_type: str
    mutual_info: Optional[float]
    f_score: Optional[float]
    correlation: Optional[float]
    chi2_score: Optional[float]
    combined_score: float
    rank: int


class ImportanceAnalyzer:
    """Analisador de importância de variáveis."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def analyze(
        self,
        df: pd.DataFrame,
        target: str,
        numeric_cols: List[str],
        categorical_cols: List[str],
        target_type: str = "auto"
    ) -> Dict[str, any]:
        """
        Analisa importância de todas as variáveis em relação ao target.

        Args:
            df: DataFrame com os dados
            target: Nome da coluna target
            numeric_cols: Lista de colunas numéricas
            categorical_cols: Lista de colunas categóricas
            target_type: "classification", "regression" ou "auto"

        Returns:
            Dicionário com rankings e scores
        """
        if not SKLEARN_AVAILABLE:
            return self._analyze_without_sklearn(df, target, numeric_cols, categorical_cols)

        # Determinar tipo do target
        if target_type == "auto":
            target_type = self._determine_target_type(df[target])

        # Preparar dados
        mask = df[target].notna()
        data = df.loc[mask].copy()

        # Codificar target se necessário
        y = data[target]
        if target_type == "classification" or not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            y = y.values

        results = {
            "target": target,
            "target_type": target_type,
            "numeric_importance": [],
            "categorical_importance": [],
            "combined_ranking": []
        }

        # Analisar numéricas
        if numeric_cols:
            results["numeric_importance"] = self._analyze_numeric(
                data, numeric_cols, y, target_type
            )

        # Analisar categóricas
        if categorical_cols:
            results["categorical_importance"] = self._analyze_categorical(
                data, categorical_cols, y, target_type
            )

        # Ranking combinado
        all_features = results["numeric_importance"] + results["categorical_importance"]
        all_features.sort(key=lambda x: x.combined_score, reverse=True)

        for i, feat in enumerate(all_features):
            feat.rank = i + 1

        results["combined_ranking"] = all_features

        return results

    def _determine_target_type(self, series: pd.Series) -> str:
        """Determina tipo do target."""
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 10:
                return "classification"
            return "regression"
        return "classification"

    def _analyze_numeric(
        self,
        data: pd.DataFrame,
        columns: List[str],
        y: np.ndarray,
        target_type: str
    ) -> List[FeatureImportance]:
        """Analisa importância de features numéricas."""
        results = []

        # Preparar matriz X
        X = data[columns].copy()

        # Imputar valores ausentes com mediana
        for col in columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        X_values = X.values

        # Normalizar para algumas métricas
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_values)

        # Mutual Information
        try:
            if target_type == "classification":
                mi_scores = mutual_info_classif(
                    X_scaled, y, random_state=self.random_state
                )
            else:
                mi_scores = mutual_info_regression(
                    X_scaled, y, random_state=self.random_state
                )
        except:
            mi_scores = np.zeros(len(columns))

        # F-score (ANOVA)
        try:
            if target_type == "classification":
                f_scores, _ = f_classif(X_values, y)
            else:
                f_scores, _ = f_regression(X_values, y)
            f_scores = np.nan_to_num(f_scores, nan=0)
        except:
            f_scores = np.zeros(len(columns))

        # Correlação (apenas para regressão)
        if target_type == "regression":
            corr_scores = []
            for col in columns:
                try:
                    corr, _ = stats.pearsonr(data[col].fillna(data[col].median()), y)
                    corr_scores.append(abs(corr))
                except:
                    corr_scores.append(0)
            corr_scores = np.array(corr_scores)
        else:
            corr_scores = np.zeros(len(columns))

        # Normalizar scores para 0-1
        mi_norm = self._normalize(mi_scores)
        f_norm = self._normalize(f_scores)
        corr_norm = self._normalize(corr_scores)

        # Score combinado
        for i, col in enumerate(columns):
            if target_type == "classification":
                combined = 0.5 * mi_norm[i] + 0.5 * f_norm[i]
            else:
                combined = 0.4 * mi_norm[i] + 0.3 * f_norm[i] + 0.3 * corr_norm[i]

            results.append(FeatureImportance(
                feature=col,
                feature_type="numeric",
                mutual_info=mi_scores[i],
                f_score=f_scores[i],
                correlation=corr_scores[i] if target_type == "regression" else None,
                chi2_score=None,
                combined_score=combined,
                rank=0
            ))

        return results

    def _analyze_categorical(
        self,
        data: pd.DataFrame,
        columns: List[str],
        y: np.ndarray,
        target_type: str
    ) -> List[FeatureImportance]:
        """Analisa importância de features categóricas."""
        results = []

        for col in columns:
            # Codificar feature
            le = LabelEncoder()
            try:
                x_encoded = le.fit_transform(data[col].fillna("__MISSING__").astype(str))
            except:
                continue

            x_encoded = x_encoded.reshape(-1, 1)

            # Mutual Information
            try:
                if target_type == "classification":
                    mi = mutual_info_classif(x_encoded, y, discrete_features=True,
                                             random_state=self.random_state)[0]
                else:
                    mi = mutual_info_regression(x_encoded, y, discrete_features=True,
                                               random_state=self.random_state)[0]
            except:
                mi = 0

            # Chi2 (apenas para classificação)
            if target_type == "classification":
                try:
                    # Chi2 requer valores não negativos
                    chi2_score, _ = chi2(x_encoded, y)
                    chi2_score = chi2_score[0]
                except:
                    chi2_score = 0
            else:
                chi2_score = None

            # Cramér's V para correlação
            try:
                contingency = pd.crosstab(data[col].fillna("__MISSING__"), y)
                chi2_stat, _, _, _ = stats.chi2_contingency(contingency)
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
            except:
                cramers_v = 0

            results.append(FeatureImportance(
                feature=col,
                feature_type="categorical",
                mutual_info=mi,
                f_score=None,
                correlation=cramers_v,
                chi2_score=chi2_score,
                combined_score=0,  # Será calculado depois
                rank=0
            ))

        # Normalizar e calcular score combinado
        if results:
            mi_scores = np.array([r.mutual_info for r in results])
            mi_norm = self._normalize(mi_scores)

            corr_scores = np.array([r.correlation for r in results])
            corr_norm = self._normalize(corr_scores)

            for i, r in enumerate(results):
                r.combined_score = 0.6 * mi_norm[i] + 0.4 * corr_norm[i]

        return results

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normaliza scores para range 0-1."""
        scores = np.nan_to_num(scores, nan=0)
        min_val = scores.min()
        max_val = scores.max()

        if max_val - min_val == 0:
            return np.zeros_like(scores)

        return (scores - min_val) / (max_val - min_val)

    def _analyze_without_sklearn(
        self,
        df: pd.DataFrame,
        target: str,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> Dict[str, any]:
        """Análise básica sem sklearn."""
        results = {
            "target": target,
            "target_type": "unknown",
            "numeric_importance": [],
            "categorical_importance": [],
            "combined_ranking": [],
            "warning": "sklearn não disponível. Usando análise básica."
        }

        # Análise básica com correlação
        y = df[target].dropna()

        for col in numeric_cols:
            if col != target:
                try:
                    mask = df[col].notna() & df[target].notna()
                    corr, _ = stats.pearsonr(df.loc[mask, col], df.loc[mask, target])
                    results["numeric_importance"].append(FeatureImportance(
                        feature=col,
                        feature_type="numeric",
                        mutual_info=None,
                        f_score=None,
                        correlation=abs(corr),
                        chi2_score=None,
                        combined_score=abs(corr),
                        rank=0
                    ))
                except:
                    pass

        # Ranking
        all_features = results["numeric_importance"]
        all_features.sort(key=lambda x: x.combined_score, reverse=True)
        for i, feat in enumerate(all_features):
            feat.rank = i + 1
        results["combined_ranking"] = all_features

        return results

    def get_summary_dataframe(self, results: Dict[str, any]) -> pd.DataFrame:
        """Converte resultados para DataFrame resumo."""
        ranking = results.get("combined_ranking", [])

        if not ranking:
            return pd.DataFrame()

        rows = []
        for r in ranking:
            rows.append({
                "Rank": r.rank,
                "Feature": r.feature,
                "Tipo": r.feature_type,
                "Mutual Info": r.mutual_info,
                "F-Score": r.f_score,
                "Correlação": r.correlation,
                "Chi2": r.chi2_score,
                "Score Combinado": r.combined_score
            })

        return pd.DataFrame(rows)
