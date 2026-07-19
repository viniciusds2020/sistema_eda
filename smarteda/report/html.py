"""Self-contained interactive HTML report powered by Plotly."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


class InteractiveHTMLReport:
    """Generate a self-contained HTML report from SmartEDA results."""

    def generate(
        self,
        results: dict[str, Any],
        df: pd.DataFrame,
        output_path: str,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        overview = results.get("overview", {})
        diagnostics = results.get("quality_diagnostics", {})
        comparison = results.get("train_test_profile", {})
        figures = self._figures(df, results)
        figure_html = []
        for index, figure in enumerate(figures):
            figure_html.append(
                pio.to_html(
                    figure,
                    full_html=False,
                    include_plotlyjs=True if index == 0 else False,
                    config={"responsive": True, "displaylogo": False},
                )
            )

        cards = [
            ("Linhas", overview.get("n_rows", len(df))),
            ("Colunas", overview.get("n_cols", len(df.columns))),
            ("Ausentes", f"{overview.get('total_missing_pct', 0):.1%}"),
            ("Duplicadas", overview.get("duplicate_rows", 0)),
            (
                "Possíveis IDs",
                diagnostics.get("summary", {}).get("possible_id_columns", 0),
            ),
            (
                "Alertas de leakage",
                diagnostics.get("summary", {}).get("possible_leakage_warnings", 0),
            ),
        ]

        body = [
            "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>SmartEDA — {escape(str(results.get('dataset_name', 'Dataset')))}</title>",
            self._styles(),
            "</head><body><main>",
            f"<h1>SmartEDA <span>{escape(str(results.get('dataset_name', 'Dataset')))}</span></h1>",
            "<p class='lead'>Relatório exploratório interativo, estatístico e reproduzível.</p>",
            "<section class='cards'>",
        ]
        for label, value in cards:
            body.append(
                f"<article class='card'><small>{escape(str(label))}</small>"
                f"<strong>{escape(str(value))}</strong></article>"
            )
        body.extend(["</section>", self._quality_section(diagnostics)])

        if comparison:
            body.append(self._comparison_section(comparison))
        if results.get("statistical_drift_tests"):
            body.append(self._statistical_section(results["statistical_drift_tests"]))
        if results.get("target_conditioned_drift"):
            body.append(self._conditioned_section(results["target_conditioned_drift"]))
        if results.get("longitudinal_monitoring"):
            body.append(self._longitudinal_section(results["longitudinal_monitoring"]))
        if results.get("preprocessing_diagnostics"):
            body.append(self._preprocessing_section(results["preprocessing_diagnostics"]))
        if results.get("time_series_diagnostics", {}).get("summary", {}).get("time_columns", 0):
            body.append(self._time_series_section(results["time_series_diagnostics"]))
        if results.get("ai_insights"):
            body.append(self._insight_section(results["ai_insights"]))

        body.append("<section><h2>Visualizações interativas</h2>")
        body.extend(figure_html)
        body.extend(
            [
                "</section>",
                "<footer>Gerado pelo SmartEDA. Alertas estatísticos exigem validação de domínio.</footer>",
                "</main></body></html>",
            ]
        )
        path.write_text("".join(body), encoding="utf-8")

    def _figures(self, df: pd.DataFrame, results: dict[str, Any]) -> list[go.Figure]:
        figures: list[go.Figure] = []
        comparison = results.get("train_test_profile", {})

        missing = df.isna().mean().sort_values(ascending=False)
        if (missing > 0).any():
            data = missing[missing > 0].reset_index()
            data.columns = ["column", "missing_rate"]
            figures.append(
                px.bar(
                    data,
                    x="column",
                    y="missing_rate",
                    title="Taxa de valores ausentes",
                )
            )

        numeric = list(df.select_dtypes(include=np.number).columns[:8])
        if numeric:
            melted = df[numeric].melt(var_name="variable", value_name="_smarteda_value")
            figures.append(
                px.histogram(
                    melted,
                    x="_smarteda_value",
                    facet_col="variable",
                    facet_col_wrap=2,
                    title="Distribuições numéricas",
                )
            )
        if len(numeric) >= 2:
            corr = df[numeric].corr()
            figures.append(
                px.imshow(
                    corr,
                    text_auto=".2f",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                    title="Correlação numérica",
                )
            )

        categorical = list(df.select_dtypes(include=["object", "category", "bool"]).columns[:6])
        for column in categorical:
            counts = (
                df[column]
                .fillna("__MISSING__")
                .astype(str)
                .value_counts()
                .head(15)
                .rename_axis("category")
                .reset_index(name="count")
            )
            figures.append(
                px.bar(
                    counts,
                    x="category",
                    y="count",
                    title=f"Categorias — {column}",
                )
            )

        if comparison.get("features"):
            drift = pd.DataFrame(comparison["features"]).sort_values("drift_score", ascending=False)
            figures.append(
                px.bar(
                    drift,
                    x="column",
                    y="drift_score",
                    color="drift_level",
                    category_orders={"drift_level": ["low", "medium", "high"]},
                    color_discrete_map={
                        "low": "#22c55e",
                        "medium": "#f59e0b",
                        "high": "#ef4444",
                    },
                    title="Drift entre treino e teste",
                    hover_data=["metric", "missing_rate_delta"],
                )
            )
        conditioned = results.get("target_conditioned_drift", {})
        if conditioned.get("features"):
            conditioned_df = pd.DataFrame(conditioned["features"])
            top = conditioned_df.sort_values("drift_score", ascending=False).head(40)
            figures.append(
                px.bar(
                    top,
                    x="column",
                    y="drift_score",
                    color="target_segment",
                    barmode="group",
                    title="Drift condicionado ao target",
                    hover_data=["drift_level", "segment_train_count", "segment_test_count"],
                )
            )

        longitudinal = results.get("longitudinal_monitoring", {})
        if longitudinal.get("feature_history"):
            history = pd.DataFrame(longitudinal["feature_history"])
            top_columns = history.groupby("column")["drift_score"].max().nlargest(12).index
            history = history[history["column"].isin(top_columns)]
            figures.append(
                px.line(
                    history,
                    x="window",
                    y="drift_score",
                    color="column",
                    markers=True,
                    category_orders={
                        "window": [row["window"] for row in longitudinal.get("windows", [])]
                    },
                    title="Evolução longitudinal do drift",
                    hover_data=["metric", "drift_level"],
                )
            )

        return figures

    @staticmethod
    def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "<p class='muted'>Nenhum alerta encontrado.</p>"
        head = "".join(f"<th>{escape(column)}</th>" for column in columns)
        body = []
        for row in rows:
            body.append(
                "<tr>"
                + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
                + "</tr>"
            )
        return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"

    def _quality_section(self, diagnostics: dict[str, Any]) -> str:
        rows = []
        for item in diagnostics.get("constant_columns", []):
            rows.append(
                {"column": item["column"], "check": "constant", "severity": item["severity"]}
            )
        for item in diagnostics.get("possible_ids", []):
            rows.append(
                {"column": item["column"], "check": "possible_id", "severity": item["severity"]}
            )
        for item in diagnostics.get("possible_leakage", []):
            rows.append(
                {"column": item["column"], "check": item["check"], "severity": item["severity"]}
            )
        return (
            "<section><h2>Diagnóstico de qualidade</h2>"
            + self._table(rows, ["column", "check", "severity"])
            + "</section>"
        )

    def _comparison_section(self, comparison: dict[str, Any]) -> str:
        summary = comparison.get("summary", {})
        return (
            "<section><h2>Perfil treino × teste</h2>"
            f"<p>Drift alto: <strong>{summary.get('high_drift_features', 0)}</strong> · "
            f"Drift médio: <strong>{summary.get('medium_drift_features', 0)}</strong> · "
            f"Colunas comuns: <strong>{summary.get('common_columns', 0)}</strong></p>"
            "</section>"
        )

    def _statistical_section(self, result: dict[str, Any]) -> str:
        rows = sorted(
            result.get("features", []),
            key=lambda row: row.get("adjusted_pvalue", 1.0),
        )[:30]
        summary = result.get("summary", {})
        return (
            "<section><h2>Testes estatísticos de drift</h2>"
            f"<p>Significativos após {escape(str(summary.get('correction', '')))}: "
            f"<strong>{summary.get('significant_after_correction', 0)}</strong></p>"
            + self._table(
                rows,
                ["column", "test", "effect_size", "pvalue", "adjusted_pvalue", "significant"],
            )
            + "</section>"
        )

    def _conditioned_section(self, result: dict[str, Any]) -> str:
        summary = result.get("summary", {})
        return (
            "<section><h2>Drift condicionado ao target</h2>"
            f"<p>Segmentos comparados: <strong>{summary.get('segments_compared', 0)}</strong> · "
            f"Achados de drift alto: <strong>{summary.get('high_drift_findings', 0)}</strong></p>"
            + self._table(
                result.get("segments", []),
                ["target_segment", "train_count", "test_count", "status"],
            )
            + "</section>"
        )

    def _longitudinal_section(self, result: dict[str, Any]) -> str:
        summary = result.get("summary", {})
        return (
            "<section><h2>Monitoramento longitudinal</h2>"
            f"<p>Janelas: <strong>{summary.get('windows', 0)}</strong> · "
            f"Com drift alto: <strong>{summary.get('windows_with_high_drift', 0)}</strong></p>"
            + self._table(
                result.get("windows", []),
                ["window", "rows", "high_drift_features", "medium_drift_features"],
            )
            + "</section>"
        )

    def _insight_section(self, result: dict[str, Any]) -> str:
        findings = "".join(f"<li>{escape(str(item))}</li>" for item in result.get("findings", []))
        recommendations = "".join(
            f"<li>{escape(str(item))}</li>" for item in result.get("recommendations", [])
        )
        provider = escape(str(result.get("provider", "rules")))
        model = escape(str(result.get("model") or "deterministic"))
        privacy = escape(str(result.get("privacy", "")))
        return (
            "<section class='agent'><div class='agent-title'>"
            "<div><small>SMARTEDA INSIGHT AGENT</small><h2>Leitura assistida do relatório</h2></div>"
            f"<span class='badge'>{provider} · {model}</span></div>"
            f"<p class='agent-summary'>{escape(str(result.get('executive_summary', '')))}</p>"
            "<div class='agent-grid'><article><h3>Achados</h3>"
            f"<ol>{findings}</ol></article><article><h3>Recomendações</h3>"
            f"<ol>{recommendations}</ol></article></div>"
            f"<p class='privacy'>🔒 {privacy}</p></section>"
        )

    def _preprocessing_section(self, result: dict[str, Any]) -> str:
        summary = result.get("summary", {})
        missing = sorted(
            result.get("missing_data", []),
            key=lambda row: row.get("missing_rate", 0),
            reverse=True,
        )[:30]
        outliers = sorted(
            result.get("outliers", []),
            key=lambda row: row.get("outlier_rate", 0),
            reverse=True,
        )[:30]
        normality = sorted(
            result.get("normality_tests", []),
            key=lambda row: row.get("pvalue", 1),
        )[:30]
        actions = result.get("prioritized_actions", [])[:30]
        return (
            "<section><h2>Diagnóstico e sugestões de pré-processamento</h2>"
            f"<p>Ausência: <strong>{summary.get('columns_with_missing', 0)}</strong> colunas · "
            f"Outliers: <strong>{summary.get('columns_with_outliers', 0)}</strong> colunas · "
            f"Não normais: <strong>{summary.get('non_normal_numeric_columns', 0)}</strong> "
            f"· Ações: <strong>{summary.get('prioritized_actions', 0)}</strong></p>"
            "<h3>Ações priorizadas</h3>"
            + self._table(actions, ["priority", "column", "action", "evidence"])
            + "<h3>Dados ausentes e associação com target</h3>"
            + self._table(
                missing,
                [
                    "column",
                    "missing_rate",
                    "suggested_strategy",
                    "target_adjusted_pvalue",
                    "target_associated",
                ],
            )
            + "<h3>Outliers</h3>"
            + self._table(
                outliers,
                ["column", "outlier_rate", "method", "suggested_action"],
            )
            + "<h3>Testes de normalidade</h3>"
            + self._table(
                normality,
                [
                    "column",
                    "test",
                    "sample_size",
                    "pvalue",
                    "is_normal",
                    "suggested_action",
                ],
            )
            + f"<p class='muted'>{escape(str(result.get('disclaimer', '')))}</p></section>"
        )

    def _time_series_section(self, result: dict[str, Any]) -> str:
        summary = result.get("summary", {})
        axes = result.get("time_axes", [])[:20]
        signals = result.get("signals", [])[:40]
        target = result.get("target_temporal_analysis", [])[:20]
        actions = result.get("prioritized_actions", [])[:30]
        return (
            "<section><h2>Diagnóstico de séries temporais</h2>"
            f"<p>Eixos: <strong>{summary.get('time_columns', 0)}</strong> · "
            f"Sinais: <strong>{summary.get('signals_analyzed', 0)}</strong> · "
            f"Eixos com gaps: <strong>{summary.get('axes_with_gaps', 0)}</strong> · "
            f"Timestamps duplicados: <strong>{summary.get('duplicate_timestamps', 0)}</strong> · "
            f"Mudanças no target: <strong>{summary.get('target_temporal_shifts', 0)}</strong></p>"
            "<h3>Ações temporais priorizadas</h3>"
            + self._table(actions, ["priority", "time_column", "action", "evidence"])
            + "<h3>Estrutura dos eixos temporais</h3>"
            + self._table(
                axes,
                [
                    "column",
                    "count",
                    "invalid_or_missing",
                    "duplicate_timestamps",
                    "inferred_frequency",
                    "regularity_ratio",
                ],
            )
            + "<h3>Sinais, tendência, sazonalidade e estacionariedade</h3>"
            + self._table(
                signals,
                [
                    "time_column",
                    "value_column",
                    "trend_spearman",
                    "trend_pvalue",
                    "lag1_autocorrelation",
                    "seasonal_period",
                    "seasonal_autocorrelation",
                    "standardized_mean_shift",
                ],
            )
            + "<h3>Target ao longo do tempo</h3>"
            + self._table(
                target,
                [
                    "time_column",
                    "target",
                    "target_type",
                    "test",
                    "pvalue",
                    "temporal_shift",
                ],
            )
            + f"<p class='muted'>{escape(str(result.get('disclaimer', '')))}</p></section>"
        )

    @staticmethod
    def _styles() -> str:
        return """<style>
        :root{color-scheme:dark;--bg:#07111f;--panel:#0f1d2e;--line:#21344a;--text:#e8f0f8;--muted:#8fa6bd;--accent:#38bdf8}
        *{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#07111f,#0b1728);color:var(--text);font:15px Inter,system-ui,sans-serif}
        main{max-width:1240px;margin:auto;padding:40px 24px}h1{font-size:36px;margin:0}h1 span{color:var(--accent);font-weight:500}
        h2{margin-top:0}.lead,.muted,footer{color:var(--muted)}section{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:22px;margin:22px 0}
        .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;background:none;border:0;padding:0}
        .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px}.card small{display:block;color:var(--muted)}.card strong{display:block;font-size:25px;margin-top:6px}
        .table-wrap{overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px}th{color:var(--accent)}
        .agent{border-color:#0ea5e9;background:linear-gradient(145deg,#0f1d2e,#10283c)}
        .agent-title{display:flex;align-items:center;justify-content:space-between;gap:16px}.agent-title small,.privacy{color:var(--accent)}
        .badge{border:1px solid var(--accent);border-radius:999px;padding:7px 11px}.agent-summary{font-size:18px;line-height:1.6}
        .agent-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.agent-grid article{background:#091727;border-radius:14px;padding:16px}.agent-grid li{margin:9px 0;line-height:1.5}
        footer{text-align:center;padding:26px}</style>"""
