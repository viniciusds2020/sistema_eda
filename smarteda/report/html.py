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
        figures = self._figures(df, comparison)
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

    def _figures(
        self, df: pd.DataFrame, comparison: dict[str, Any]
    ) -> list[go.Figure]:
        figures: list[go.Figure] = []

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
            melted = df[numeric].melt(var_name="variable", value_name="value")
            figures.append(
                px.histogram(
                    melted,
                    x="value",
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

        categorical = list(
            df.select_dtypes(include=["object", "category", "bool"]).columns[:6]
        )
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
            drift = pd.DataFrame(comparison["features"]).sort_values(
                "drift_score", ascending=False
            )
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
            rows.append({"column": item["column"], "check": "constant", "severity": item["severity"]})
        for item in diagnostics.get("possible_ids", []):
            rows.append({"column": item["column"], "check": "possible_id", "severity": item["severity"]})
        for item in diagnostics.get("possible_leakage", []):
            rows.append({"column": item["column"], "check": item["check"], "severity": item["severity"]})
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
        footer{text-align:center;padding:26px}</style>"""
