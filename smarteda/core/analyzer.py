"""Classe principal SmartEDA."""

from typing import Any, Dict, Optional

import pandas as pd
from smarteda.analysis.categorical import CategoricalAnalyzer
from smarteda.analysis.conditioned import conditioned_frame, target_conditioned_drift
from smarteda.analysis.correlation import CorrelationAnalyzer
from smarteda.analysis.drift import profile_frame
from smarteda.analysis.drift import profile_train_test as build_train_test_profile
from smarteda.analysis.importance import ImportanceAnalyzer
from smarteda.analysis.monitoring import longitudinal_frame, monitor_windows
from smarteda.analysis.numeric import NumericAnalyzer
from smarteda.analysis.quality import detect_quality_issues
from smarteda.analysis.statistical_tests import distribution_tests, tests_frame
from smarteda.analysis.target import TargetAnalyzer
from smarteda.analysis.temporal import TemporalAnalyzer
from smarteda.core.adapters import to_pandas
from smarteda.core.type_inference import DataType, TypeInference
from smarteda.report.generator import ReportGenerator

from smarteda.core.config import Config
from smarteda.insights import InsightAgent, InsightContextBuilder
from smarteda.report.html import InteractiveHTMLReport


class SmartEDA:
    """
    SmartEDA - Análise Exploratória de Dados Inteligente.

    Replica as funcionalidades do pacote SmartEDA do R em Python.

    Exemplo de uso:
        >>> eda = SmartEDA(df)
        >>> eda.analyze()
        >>> eda.generate_report("relatorio.md")

        >>> # Com variável target
        >>> eda = SmartEDA(df, target="target_column")
        >>> eda.analyze()
        >>> eda.generate_report("relatorio.md")
    """

    def __init__(
        self,
        df: Any,
        target: Optional[str] = None,
        config: Optional[Config] = None,
        dataset_name: str = "Dataset",
    ):
        """
        Inicializa o SmartEDA.

        Args:
            df: DataFrame pandas com os dados
            target: Nome da coluna target (opcional)
            config: Configurações personalizadas (opcional)
            dataset_name: Nome do dataset para o relatório
        """
        self.config = config or Config()
        self.df = to_pandas(
            df,
            max_rows=self.config.sample_size,
            random_state=self.config.random_state,
        )
        self.target = target
        self.dataset_name = dataset_name

        # Aplicar amostragem se configurado
        if self.config.sample_size and len(self.df) > self.config.sample_size:
            self.df = self.df.sample(
                n=self.config.sample_size, random_state=self.config.random_state
            )

        # Inicializar analisadores
        self.type_inference = TypeInference(
            categorical_threshold=self.config.categorical_threshold,
            id_unique_ratio=self.config.id_unique_ratio,
        )

        self.numeric_analyzer = NumericAnalyzer(
            percentiles=self.config.percentiles,
            iqr_multiplier=self.config.outlier_iqr_multiplier,
            zscore_threshold=self.config.outlier_zscore_threshold,
        )

        self.categorical_analyzer = CategoricalAnalyzer(
            top_n=self.config.top_n_categories, rare_threshold=self.config.rare_category_threshold
        )

        self.temporal_analyzer = TemporalAnalyzer()

        self.correlation_analyzer = CorrelationAnalyzer(
            min_correlation=self.config.correlation_threshold
        )

        self.target_analyzer = TargetAnalyzer(
            classification_threshold=self.config.classification_threshold
        )

        self.importance_analyzer = ImportanceAnalyzer(random_state=self.config.random_state)

        self.report_generator = ReportGenerator(
            include_plots=self.config.include_plots,
            plot_format=self.config.plot_format,
            plot_dpi=self.config.plot_dpi,
            max_categories_plot=self.config.max_categories_plot,
        )

        # Resultados
        self.results: Dict[str, Any] = {}
        self._analyzed = False

    def analyze(self) -> Dict[str, Any]:
        """
        Executa análise exploratória completa.

        Returns:
            Dicionário com todos os resultados da análise
        """
        print("[SmartEDA] Iniciando analise exploratoria...")

        # 1. Inferência de tipos
        print("  [1/8] Inferindo tipos de dados...")
        self.columns_info = self.type_inference.infer_types(self.df)

        # Aplicar conversões
        self.df = self.type_inference.apply_conversions(self.df, self.columns_info)

        # Separar colunas por tipo
        self.numeric_cols = self.type_inference.get_columns_by_type(
            self.columns_info, [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]
        )
        self.categorical_cols = self.type_inference.get_columns_by_type(
            self.columns_info, [DataType.CATEGORICAL, DataType.BINARY]
        )
        self.temporal_cols = self.type_inference.get_columns_by_type(
            self.columns_info, [DataType.DATETIME]
        )

        # 2. Visão geral
        print("  [2/8] Calculando visao geral...")
        self.results["overview"] = self._compute_overview()
        self.results["columns_info"] = self.columns_info
        self.results["dataset_name"] = self.dataset_name
        self.results["quality_diagnostics"] = detect_quality_issues(
            self.df,
            target=self.target,
            id_unique_ratio=self.config.id_unique_ratio,
        )

        # 3. Análise numérica
        if self.numeric_cols:
            print(f"  [3/8] Analisando {len(self.numeric_cols)} variaveis numericas...")
            self.results["numeric_stats"] = self.numeric_analyzer.analyze(
                self.df, self.numeric_cols
            )

        # 4. Análise categórica
        if self.categorical_cols:
            print(f"  [4/8] Analisando {len(self.categorical_cols)} variaveis categoricas...")
            self.results["categorical_stats"] = self.categorical_analyzer.analyze(
                self.df, self.categorical_cols
            )

        # 5. Análise temporal
        if self.temporal_cols:
            print(f"  [5/8] Analisando {len(self.temporal_cols)} variaveis temporais...")
            self.results["temporal_stats"] = self.temporal_analyzer.analyze(
                self.df, self.temporal_cols
            )

        # 6. Correlações
        print("  [6/8] Calculando correlacoes...")
        self.results["correlations"] = self._compute_correlations()

        # 7. Análise com target
        if self.target and self.target in self.df.columns:
            print(f"  🎯 Analisando relação com target '{self.target}'...")
            # Remover target das listas de features
            numeric_features = [c for c in self.numeric_cols if c != self.target]
            categorical_features = [c for c in self.categorical_cols if c != self.target]

            self.results["target_analysis"] = self.target_analyzer.analyze(
                self.df, self.target, numeric_features, categorical_features
            )

            # 8. Importância de variáveis
            print("  ⭐ Calculando importância de variáveis...")
            self.results["importance"] = self.importance_analyzer.analyze(
                self.df, self.target, numeric_features, categorical_features
            )

        self._analyzed = True
        print("✅ Análise concluída!")

        return self.results

    def _compute_overview(self) -> Dict[str, Any]:
        """Calcula visão geral do dataset."""
        n_rows, n_cols = self.df.shape

        # Contagem por tipo
        n_numeric = len(self.numeric_cols)
        n_categorical = len(self.categorical_cols)
        n_temporal = len(self.temporal_cols)

        # Dados ausentes
        total_cells = n_rows * n_cols
        total_missing = self.df.isna().sum().sum()
        total_missing_pct = total_missing / total_cells if total_cells > 0 else 0

        # Duplicadas
        duplicate_rows = self.df.duplicated().sum()

        # Memória
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_numeric": n_numeric,
            "n_categorical": n_categorical,
            "n_temporal": n_temporal,
            "total_missing": total_missing,
            "total_missing_pct": total_missing_pct,
            "duplicate_rows": duplicate_rows,
            "memory_mb": memory_mb,
        }

    def _compute_correlations(self) -> Dict[str, Any]:
        """Calcula todas as correlações."""
        correlations = {}

        # Numéricas
        if len(self.numeric_cols) >= 2:
            corr_matrix, p_matrix, significant = self.correlation_analyzer.numeric_correlation(
                self.df, self.numeric_cols, method="pearson"
            )
            correlations["numeric_matrix"] = corr_matrix
            correlations["numeric_pvalues"] = p_matrix
            correlations["numeric_significant"] = significant

        # Categóricas
        if len(self.categorical_cols) >= 2:
            cramer_matrix, significant = self.correlation_analyzer.categorical_correlation(
                self.df, self.categorical_cols
            )
            correlations["categorical_matrix"] = cramer_matrix
            correlations["categorical_significant"] = significant

        # Mistas
        if self.numeric_cols and self.categorical_cols:
            eta_matrix, significant = self.correlation_analyzer.mixed_correlation(
                self.df, self.numeric_cols, self.categorical_cols
            )
            correlations["mixed_matrix"] = eta_matrix
            correlations["mixed_significant"] = significant

        return correlations

    def profile_train_test(self, test_df: Any, bins: int = 10) -> pd.DataFrame:
        """Compare this analysis dataset with a test or reference dataset.

        The comparison includes schema changes, missing-rate deltas, PSI for
        numeric features, Jensen-Shannon divergence for categorical features,
        and unseen-category rates.
        """
        comparison = build_train_test_profile(
            self.df,
            test_df,
            target=self.target,
            bins=bins,
            max_rows=self.config.sample_size,
            random_state=self.config.random_state,
        )
        self.results["train_test_profile"] = comparison
        return profile_frame(comparison)

    def run_distribution_tests(
        self,
        test_df: Any,
        *,
        correction: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run train/test distribution tests with adjusted p-values."""
        result = distribution_tests(
            self.df,
            test_df,
            target=self.target,
            alpha=alpha or self.config.statistical_alpha,
            correction=correction or self.config.pvalue_correction,
            max_rows=self.config.sample_size,
            random_state=self.config.random_state,
        )
        self.results["statistical_drift_tests"] = result
        return tests_frame(result)

    def profile_target_conditioned(
        self,
        test_df: Any,
        *,
        target_bins: Optional[int] = None,
        min_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Profile drift inside target classes or target quantile bands."""
        if not self.target:
            raise ValueError("A target must be configured for conditioned drift.")
        result = target_conditioned_drift(
            self.df,
            test_df,
            target=self.target,
            bins=self.config.drift_bins,
            target_bins=target_bins or self.config.target_bins,
            classification_threshold=self.config.classification_threshold,
            min_samples=min_samples or self.config.min_segment_size,
            max_rows=self.config.sample_size,
            random_state=self.config.random_state,
        )
        self.results["target_conditioned_drift"] = result
        return conditioned_frame(result)

    def monitor_windows(self, windows: Dict[str, Any]) -> pd.DataFrame:
        """Monitor multiple named windows against the analysis dataset."""
        result = monitor_windows(
            self.df,
            windows,
            target=self.target,
            bins=self.config.drift_bins,
            max_rows=self.config.sample_size,
            random_state=self.config.random_state,
        )
        self.results["longitudinal_monitoring"] = result
        return longitudinal_frame(result)

    def build_insight_context(self) -> Dict[str, Any]:
        """Return a compact context containing aggregate results only."""
        if not self._analyzed:
            self.analyze()
        return InsightContextBuilder().build(self.results)

    def generate_insights(
        self,
        *,
        provider: str = "rules",
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate local or Groq/Llama insights from aggregate statistics."""
        if not self._analyzed:
            self.analyze()
        insights = InsightAgent(
            provider=provider,
            model=model,
            api_key=api_key,
        ).summarize(self.results)
        self.results["ai_insights"] = insights
        return insights

    def ask(
        self,
        question: str,
        *,
        provider: str = "rules",
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ) -> str:
        """Ask a question using only compact statistical results."""
        if not self._analyzed:
            self.analyze()
        return InsightAgent(
            provider=provider,
            model=model,
            api_key=api_key,
        ).ask(question, self.results)

    def generate_html_report(
        self,
        output_path: str = "eda_report.html",
        *,
        enable_agent: bool = False,
        agent_provider: str = "rules",
        agent_model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ) -> None:
        """Generate a self-contained report with optional precomputed insights."""
        if not self._analyzed:
            self.analyze()

        if enable_agent:
            self.generate_insights(
                provider=agent_provider,
                model=agent_model,
                api_key=api_key,
            )

        InteractiveHTMLReport().generate(
            self.results,
            self.df,
            output_path,
        )

    def generate_report(
        self, output_path: str = "eda_report.md", include_html: bool = True
    ) -> None:
        """
        Gera relatório em Markdown.

        Args:
            output_path: Caminho do arquivo de saída
            include_html: Se True, inclui elementos HTML para design moderno
        """
        if not self._analyzed:
            print("⚠️ Executando análise primeiro...")
            self.analyze()

        print(f"📝 Gerando relatório em '{output_path}'...")
        self.report_generator.generate(self.results, output_path, self.df, include_html)
        print("✅ Relatório salvo!")

    def get_numeric_summary(self) -> pd.DataFrame:
        """Retorna resumo das variáveis numéricas como DataFrame."""
        if "numeric_stats" not in self.results:
            return pd.DataFrame()
        return self.numeric_analyzer.get_summary_dataframe(self.results["numeric_stats"])

    def get_categorical_summary(self) -> pd.DataFrame:
        """Retorna resumo das variáveis categóricas como DataFrame."""
        if "categorical_stats" not in self.results:
            return pd.DataFrame()
        return self.categorical_analyzer.get_summary_dataframe(self.results["categorical_stats"])

    def get_temporal_summary(self) -> pd.DataFrame:
        """Retorna resumo das variáveis temporais como DataFrame."""
        if "temporal_stats" not in self.results:
            return pd.DataFrame()
        return self.temporal_analyzer.get_summary_dataframe(self.results["temporal_stats"])

    def get_correlation_summary(self) -> pd.DataFrame:
        """Retorna resumo das correlações significativas."""
        correlations = self.results.get("correlations", {})
        return self.correlation_analyzer.get_all_significant_correlations(
            correlations.get("numeric_significant", []),
            correlations.get("categorical_significant", []),
            correlations.get("mixed_significant", []),
        )

    def get_target_summary(self) -> pd.DataFrame:
        """Retorna resumo da análise com target."""
        if "target_analysis" not in self.results:
            return pd.DataFrame()
        return self.target_analyzer.get_summary_dataframe(self.results["target_analysis"])

    def get_importance_summary(self) -> pd.DataFrame:
        """Retorna ranking de importância de variáveis."""
        if "importance" not in self.results:
            return pd.DataFrame()
        return self.importance_analyzer.get_summary_dataframe(self.results["importance"])

    def __repr__(self) -> str:
        status = "analyzed" if self._analyzed else "not analyzed"
        return f"SmartEDA(rows={len(self.df)}, cols={len(self.df.columns)}, target={self.target}, status={status})"
