"""Gerador de relatórios em Markdown."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

from smarteda.report.styles import Styles, AlertLevel
from smarteda.core.type_inference import DataType

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class ReportGenerator:
    """Gerador de relatórios EDA em Markdown."""

    def __init__(
        self,
        include_plots: bool = True,
        plot_format: str = "png",
        plot_dpi: int = 100,
        max_categories_plot: int = 15
    ):
        self.include_plots = include_plots and PLOTTING_AVAILABLE
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.max_categories_plot = max_categories_plot
        self.styles = Styles()
        self.plots_dir = None

    def generate(
        self,
        analysis_results: Dict[str, Any],
        output_path: str,
        df: pd.DataFrame = None,
        include_html: bool = True
    ) -> None:
        """
        Gera relatório completo.

        Args:
            analysis_results: Resultados da análise do SmartEDA
            output_path: Caminho do arquivo de saída (.md)
            df: DataFrame original (para gráficos)
            include_html: Se True, gera versão com HTML também
        """
        output_path = Path(output_path)

        # Criar diretório de plots se necessário
        if self.include_plots and df is not None:
            self.plots_dir = output_path.parent / f"{output_path.stem}_plots"
            self.plots_dir.mkdir(exist_ok=True)

        # Gerar conteúdo
        content_plain = self._generate_content(analysis_results, df, use_html=False)
        content_html = self._generate_content(analysis_results, df, use_html=True)

        # Salvar relatório principal (com HTML)
        if include_html:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content_html)

            # Salvar versão plain também
            plain_path = output_path.parent / f"{output_path.stem}_plain.md"
            with open(plain_path, 'w', encoding='utf-8') as f:
                f.write(content_plain)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content_plain)

    def _generate_content(
        self,
        results: Dict[str, Any],
        df: pd.DataFrame,
        use_html: bool
    ) -> str:
        """Gera conteúdo do relatório."""
        sections = []

        # Cabeçalho
        sections.append(self._header(results, use_html))

        # Sumário executivo
        sections.append(self._executive_summary(results, use_html))

        # Visão geral dos dados
        sections.append(self._data_overview(results, use_html))

        # Inferência de tipos
        sections.append(self._type_inference_section(results, use_html))

        # Análise numérica
        if results.get("numeric_stats"):
            sections.append(self._numeric_section(results, df, use_html))

        # Análise categórica
        if results.get("categorical_stats"):
            sections.append(self._categorical_section(results, df, use_html))

        # Análise temporal
        if results.get("temporal_stats"):
            sections.append(self._temporal_section(results, use_html))

        # Correlações
        if results.get("correlations"):
            sections.append(self._correlation_section(results, df, use_html))

        # Análise com target
        if results.get("target_analysis"):
            sections.append(self._target_section(results, df, use_html))

        # Importância de variáveis
        if results.get("importance"):
            sections.append(self._importance_section(results, use_html))

        # Alertas e insights
        sections.append(self._insights_section(results, use_html))

        # Rodapé
        sections.append(self._footer(use_html))

        return "\n\n".join(sections)

    def _header(self, results: Dict[str, Any], use_html: bool) -> str:
        """Gera cabeçalho do relatório."""
        dataset_name = results.get("dataset_name", "Dataset")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# {Styles.icon('chart')} Relatório de Análise Exploratória

**Dataset:** {dataset_name}
**Gerado em:** {timestamp}
**Gerado por:** SmartEDA Python

---
"""
        return content

    def _executive_summary(self, results: Dict[str, Any], use_html: bool) -> str:
        """Gera sumário executivo."""
        overview = results.get("overview", {})

        n_rows = overview.get("n_rows", 0)
        n_cols = overview.get("n_cols", 0)
        n_numeric = overview.get("n_numeric", 0)
        n_categorical = overview.get("n_categorical", 0)
        n_temporal = overview.get("n_temporal", 0)
        missing_pct = overview.get("total_missing_pct", 0)
        duplicates = overview.get("duplicate_rows", 0)

        if use_html:
            cards = f"""
<div style="display:flex;flex-wrap:wrap;gap:10px;margin:20px 0;">
{Styles.metric_card("Linhas", f"{n_rows:,}")}
{Styles.metric_card("Colunas", str(n_cols))}
{Styles.metric_card("Numéricas", str(n_numeric))}
{Styles.metric_card("Categóricas", str(n_categorical))}
{Styles.metric_card("Temporais", str(n_temporal))}
{Styles.metric_card("Ausentes", f"{missing_pct:.1%}")}
</div>
"""
        else:
            cards = f"""
| Métrica | Valor |
|---------|-------|
| Linhas | {n_rows:,} |
| Colunas | {n_cols} |
| Numéricas | {n_numeric} |
| Categóricas | {n_categorical} |
| Temporais | {n_temporal} |
| % Ausentes | {missing_pct:.1%} |
| Duplicadas | {duplicates:,} |
"""

        content = f"""## {Styles.icon('info')} Sumário Executivo

{cards}
"""

        # Alertas importantes
        alerts = []
        if missing_pct > 0.1:
            alerts.append(Styles.alert(f"Alto percentual de dados ausentes ({missing_pct:.1%})", AlertLevel.WARNING))
        if duplicates > 0:
            alerts.append(Styles.alert(f"Existem {duplicates:,} linhas duplicadas", AlertLevel.WARNING))

        if alerts:
            content += "\n### Alertas\n\n" + "\n\n".join(alerts)

        return content

    def _data_overview(self, results: Dict[str, Any], use_html: bool) -> str:
        """Gera visão geral dos dados."""
        overview = results.get("overview", {})
        columns_info = results.get("columns_info", {})

        content = f"""## {Styles.icon('table')} Visão Geral dos Dados

### Estrutura do Dataset

"""

        if columns_info:
            rows = []
            for col, info in columns_info.items():
                type_badge = self._get_type_badge(info.inferred_type, use_html)
                rows.append({
                    "Coluna": col,
                    "Tipo Original": info.original_dtype,
                    "Tipo Inferido": type_badge if use_html else info.inferred_type.value,
                    "Únicos": info.unique_count,
                    "Ausentes": f"{info.null_percentage:.1%}"
                })

            df_info = pd.DataFrame(rows)
            content += Styles.format_table(df_info, max_rows=50)

        return content

    def _type_inference_section(self, results: Dict[str, Any], use_html: bool) -> str:
        """Seção de inferência de tipos."""
        columns_info = results.get("columns_info", {})

        # Agrupar por tipo
        type_groups = {}
        for col, info in columns_info.items():
            type_name = info.inferred_type.value
            if type_name not in type_groups:
                type_groups[type_name] = []
            type_groups[type_name].append(col)

        content = f"""## {Styles.icon('category')} Inferência de Tipos

### Distribuição por Tipo

"""
        for type_name, cols in type_groups.items():
            content += f"- **{type_name}**: {len(cols)} colunas\n"

        return content

    def _numeric_section(
        self, results: Dict[str, Any], df: pd.DataFrame, use_html: bool
    ) -> str:
        """Seção de análise numérica."""
        numeric_stats = results.get("numeric_stats", {})

        content = f"""## {Styles.icon('number')} Análise de Variáveis Numéricas

### Estatísticas Descritivas

"""

        # Tabela resumo
        rows = []
        for col, stats in numeric_stats.items():
            rows.append({
                "Variável": stats.column,
                "N": stats.count,
                "Média": f"{stats.mean:.2f}" if not np.isnan(stats.mean) else "N/A",
                "Mediana": f"{stats.median:.2f}" if not np.isnan(stats.median) else "N/A",
                "Desvio": f"{stats.std:.2f}" if not np.isnan(stats.std) else "N/A",
                "Mín": f"{stats.min:.2f}" if not np.isnan(stats.min) else "N/A",
                "Máx": f"{stats.max:.2f}" if not np.isnan(stats.max) else "N/A",
                "Assimetria": f"{stats.skewness:.2f}" if not np.isnan(stats.skewness) else "N/A",
                "Outliers": stats.outliers_iqr
            })

        df_stats = pd.DataFrame(rows)
        content += Styles.format_table(df_stats)

        # Percentis detalhados
        content += "\n\n### Percentis\n\n"
        percentile_rows = []
        for col, stats in numeric_stats.items():
            row = {"Variável": col}
            for pct, val in stats.percentiles.items():
                row[pct] = f"{val:.2f}" if not np.isnan(val) else "N/A"
            percentile_rows.append(row)

        if percentile_rows:
            df_pct = pd.DataFrame(percentile_rows)
            content += Styles.format_table(df_pct)

        # Gráficos
        if self.include_plots and df is not None:
            content += self._generate_numeric_plots(df, list(numeric_stats.keys()))

        return content

    def _categorical_section(
        self, results: Dict[str, Any], df: pd.DataFrame, use_html: bool
    ) -> str:
        """Seção de análise categórica."""
        cat_stats = results.get("categorical_stats", {})

        content = f"""## {Styles.icon('category')} Análise de Variáveis Categóricas

### Resumo

"""

        rows = []
        for col, stats in cat_stats.items():
            rows.append({
                "Variável": stats.column,
                "N": stats.count,
                "Únicos": stats.unique,
                "Moda": stats.mode[:20] + "..." if len(stats.mode) > 20 else stats.mode,
                "Moda%": f"{stats.mode_pct:.1%}",
                "Entropia": f"{stats.entropy:.2f}",
                "Raros": stats.rare_count
            })

        df_stats = pd.DataFrame(rows)
        content += Styles.format_table(df_stats)

        # Top categorias para cada variável
        content += "\n\n### Top Categorias por Variável\n\n"

        for col, stats in cat_stats.items():
            if stats.top_categories:
                content += f"\n#### {col}\n\n"
                top_rows = []
                for cat in stats.top_categories[:10]:
                    top_rows.append({
                        "Categoria": cat["category"][:30],
                        "Contagem": cat["count"],
                        "Percentual": f"{cat['percentage']:.1%}"
                    })
                df_top = pd.DataFrame(top_rows)
                content += Styles.format_table(df_top) + "\n"

        # Gráficos
        if self.include_plots and df is not None:
            content += self._generate_categorical_plots(df, list(cat_stats.keys()))

        return content

    def _temporal_section(self, results: Dict[str, Any], use_html: bool) -> str:
        """Seção de análise temporal."""
        temp_stats = results.get("temporal_stats", {})

        content = f"""## {Styles.icon('calendar')} Análise de Variáveis Temporais

### Resumo

"""

        rows = []
        for col, stats in temp_stats.items():
            rows.append({
                "Variável": stats.column,
                "N": stats.count,
                "Início": str(stats.min_date)[:10] if stats.min_date else "N/A",
                "Fim": str(stats.max_date)[:10] if stats.max_date else "N/A",
                "Range": stats.range_str,
                "Tem Hora": "Sim" if stats.has_time else "Não",
                "Gaps": len(stats.gaps)
            })

        df_stats = pd.DataFrame(rows)
        content += Styles.format_table(df_stats)

        return content

    def _correlation_section(
        self, results: Dict[str, Any], df: pd.DataFrame, use_html: bool
    ) -> str:
        """Seção de correlações."""
        correlations = results.get("correlations", {})

        content = f"""## {Styles.icon('correlation')} Análise de Correlações

"""

        # Correlações numéricas significativas
        if correlations.get("numeric_significant"):
            content += "### Correlações Numéricas Significativas\n\n"
            rows = []
            for corr in correlations["numeric_significant"][:20]:
                rows.append({
                    "Variável 1": corr.var1,
                    "Variável 2": corr.var2,
                    "Correlação": f"{corr.correlation:.3f}",
                    "P-valor": f"{corr.p_value:.4f}" if corr.p_value else "N/A"
                })
            if rows:
                content += Styles.format_table(pd.DataFrame(rows)) + "\n\n"

        # Associações categóricas
        if correlations.get("categorical_significant"):
            content += "### Associações Categóricas (Cramér's V)\n\n"
            rows = []
            for corr in correlations["categorical_significant"][:20]:
                rows.append({
                    "Variável 1": corr.var1,
                    "Variável 2": corr.var2,
                    "Cramér's V": f"{corr.correlation:.3f}",
                    "P-valor": f"{corr.p_value:.4f}" if corr.p_value else "N/A"
                })
            if rows:
                content += Styles.format_table(pd.DataFrame(rows)) + "\n\n"

        # Relações mistas
        if correlations.get("mixed_significant"):
            content += "### Relações Numéricas x Categóricas (Eta²)\n\n"
            rows = []
            for corr in correlations["mixed_significant"][:20]:
                rows.append({
                    "Numérica": corr.var1,
                    "Categórica": corr.var2,
                    "Eta²": f"{corr.correlation:.3f}",
                    "P-valor": f"{corr.p_value:.4f}" if corr.p_value else "N/A"
                })
            if rows:
                content += Styles.format_table(pd.DataFrame(rows)) + "\n\n"

        # Gráfico de correlação
        if self.include_plots and df is not None and correlations.get("numeric_matrix") is not None:
            content += self._generate_correlation_plot(correlations["numeric_matrix"])

        return content

    def _target_section(
        self, results: Dict[str, Any], df: pd.DataFrame, use_html: bool
    ) -> str:
        """Seção de análise com target."""
        target_analysis = results.get("target_analysis", {})

        target_name = target_analysis.get("target_name", "Target")
        target_type = target_analysis.get("target_type", "unknown")

        content = f"""## {Styles.icon('target')} Análise com Variável Target

**Target:** {target_name}
**Tipo:** {target_type}

"""

        # Estatísticas do target
        target_stats = target_analysis.get("target_stats", {})
        if target_type == "classification":
            content += "### Distribuição das Classes\n\n"
            if "class_distribution" in target_stats:
                rows = [{"Classe": k, "Contagem": v}
                        for k, v in target_stats["class_distribution"].items()]
                content += Styles.format_table(pd.DataFrame(rows)) + "\n\n"

        # Ranking de features
        ranking = target_analysis.get("feature_ranking", [])
        if ranking:
            content += "### Ranking de Features por Importância\n\n"
            rows = []
            for i, feat in enumerate(ranking[:20], 1):
                rows.append({
                    "#": i,
                    "Feature": feat.feature,
                    "Tipo": feat.feature_type,
                    "Métrica": feat.metric_name,
                    "Valor": f"{feat.metric_value:.4f}" if feat.metric_value else "N/A",
                    "Interpretação": feat.interpretation
                })
            content += Styles.format_table(pd.DataFrame(rows))

        return content

    def _importance_section(self, results: Dict[str, Any], use_html: bool) -> str:
        """Seção de importância de variáveis."""
        importance = results.get("importance", {})
        ranking = importance.get("combined_ranking", [])

        if not ranking:
            return ""

        content = f"""## {Styles.icon('importance')} Importância de Variáveis

### Ranking Consolidado

"""

        rows = []
        for feat in ranking[:30]:
            rows.append({
                "#": feat.rank,
                "Feature": feat.feature,
                "Tipo": feat.feature_type,
                "Mutual Info": f"{feat.mutual_info:.4f}" if feat.mutual_info else "-",
                "Score": f"{feat.combined_score:.4f}"
            })

        content += Styles.format_table(pd.DataFrame(rows))

        return content

    def _insights_section(self, results: Dict[str, Any], use_html: bool) -> str:
        """Seção de insights automáticos."""
        insights = self._generate_insights(results)

        if not insights:
            return ""

        content = f"""## {Styles.icon('info')} Insights e Recomendações

"""
        for insight in insights:
            content += f"- {insight}\n"

        return content

    def _generate_insights(self, results: Dict[str, Any]) -> List[str]:
        """Gera insights automáticos baseados na análise."""
        insights = []
        overview = results.get("overview", {})

        # Missing data
        missing_pct = overview.get("total_missing_pct", 0)
        if missing_pct > 0.3:
            insights.append(f"⚠️ Alto volume de dados ausentes ({missing_pct:.1%}). Considere estratégias de imputação.")
        elif missing_pct > 0.1:
            insights.append(f"ℹ️ Dados ausentes detectados ({missing_pct:.1%}). Verifique padrões de ausência.")

        # Outliers
        numeric_stats = results.get("numeric_stats", {})
        high_outlier_cols = [
            col for col, stats in numeric_stats.items()
            if stats.outliers_iqr_pct > 0.05
        ]
        if high_outlier_cols:
            insights.append(f"⚡ Colunas com muitos outliers: {', '.join(high_outlier_cols[:5])}")

        # Assimetria
        skewed_cols = [
            col for col, stats in numeric_stats.items()
            if abs(stats.skewness) > 1
        ]
        if skewed_cols:
            insights.append(f"📊 Colunas com alta assimetria: {', '.join(skewed_cols[:5])}. Considere transformação log.")

        # Correlações fortes
        correlations = results.get("correlations", {})
        if correlations.get("numeric_significant"):
            high_corr = [c for c in correlations["numeric_significant"] if abs(c.correlation) > 0.8]
            if high_corr:
                pairs = [f"{c.var1} ↔ {c.var2}" for c in high_corr[:3]]
                insights.append(f"🔗 Variáveis altamente correlacionadas: {', '.join(pairs)}. Possível multicolinearidade.")

        # Categorias raras
        cat_stats = results.get("categorical_stats", {})
        rare_cols = [col for col, stats in cat_stats.items() if stats.rare_count > 5]
        if rare_cols:
            insights.append(f"🏷️ Colunas com muitas categorias raras: {', '.join(rare_cols[:3])}. Considere agrupamento.")

        return insights

    def _footer(self, use_html: bool) -> str:
        """Gera rodapé do relatório."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
---

*Relatório gerado automaticamente pelo SmartEDA Python em {timestamp}*
"""

    def _get_type_badge(self, data_type: DataType, use_html: bool) -> str:
        """Retorna badge para tipo de dado."""
        type_map = {
            DataType.NUMERIC_CONTINUOUS: "numeric",
            DataType.NUMERIC_DISCRETE: "numeric",
            DataType.CATEGORICAL: "categorical",
            DataType.BINARY: "binary",
            DataType.DATETIME: "temporal",
            DataType.TEXT: "text",
            DataType.ID: "id",
        }
        badge_type = type_map.get(data_type, "categorical")

        if use_html:
            return Styles.badge(badge_type)
        return Styles.badge_plain(badge_type)

    def _generate_numeric_plots(self, df: pd.DataFrame, columns: List[str]) -> str:
        """Gera gráficos para variáveis numéricas."""
        content = "\n\n### Distribuições\n\n"

        for col in columns[:10]:  # Limitar a 10 colunas
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Histograma
            axes[0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_title(f'Histograma: {col}')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequência')

            # Boxplot
            axes[1].boxplot(df[col].dropna())
            axes[1].set_title(f'Boxplot: {col}')
            axes[1].set_ylabel(col)

            plt.tight_layout()

            # Salvar
            plot_path = self.plots_dir / f"numeric_{col}.{self.plot_format}"
            plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()

            rel_path = f"{self.plots_dir.name}/{plot_path.name}"
            content += f"![{col}]({rel_path})\n\n"

        return content

    def _generate_categorical_plots(self, df: pd.DataFrame, columns: List[str]) -> str:
        """Gera gráficos para variáveis categóricas."""
        content = "\n\n### Distribuições\n\n"

        for col in columns[:10]:
            value_counts = df[col].value_counts().head(self.max_categories_plot)

            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title(f'Top Categorias: {col}')
            ax.set_xlabel('Frequência')

            plt.tight_layout()

            plot_path = self.plots_dir / f"categorical_{col}.{self.plot_format}"
            plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()

            rel_path = f"{self.plots_dir.name}/{plot_path.name}"
            content += f"![{col}]({rel_path})\n\n"

        return content

    def _generate_correlation_plot(self, corr_matrix: pd.DataFrame) -> str:
        """Gera heatmap de correlação."""
        content = "\n\n### Matriz de Correlação\n\n"

        if len(corr_matrix) > 20:
            # Limitar a 20 variáveis mais correlacionadas
            corr_matrix = corr_matrix.iloc[:20, :20]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True,
            ax=ax,
            annot_kws={'size': 8}
        )
        ax.set_title('Matriz de Correlação')

        plt.tight_layout()

        plot_path = self.plots_dir / f"correlation_matrix.{self.plot_format}"
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()

        rel_path = f"{self.plots_dir.name}/{plot_path.name}"
        content += f"![Correlação]({rel_path})\n\n"

        return content
