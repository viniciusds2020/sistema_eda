from pathlib import Path

import pandas as pd

from smarteda.report.html import InteractiveHTMLReport


def test_html_renders_insight_agent_without_exposing_credentials(tmp_path: Path):
    output = tmp_path / "report.html"
    results = {
        "dataset_name": "Credit",
        "overview": {"n_rows": 2, "n_cols": 1},
        "ai_insights": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "executive_summary": "Drift requer investigação.",
            "findings": ["Duas variáveis com drift alto."],
            "recommendations": ["Validar impacto no modelo."],
            "privacy": "Somente estatísticas agregadas.",
        },
    }
    InteractiveHTMLReport().generate(results, pd.DataFrame({"value": [1, 2]}), str(output))
    content = output.read_text(encoding="utf-8")
    assert "SMARTEDA INSIGHT AGENT" in content
    assert "Duas variáveis com drift alto" in content
    assert "GROQ_API_KEY" not in content


def test_html_renders_preprocessing_diagnostics(tmp_path: Path):
    output = tmp_path / "preprocessing.html"
    results = {
        "dataset_name": "Credit",
        "overview": {"n_rows": 10, "n_cols": 2},
        "preprocessing_diagnostics": {
            "summary": {
                "columns_with_missing": 1,
                "columns_with_outliers": 1,
                "non_normal_numeric_columns": 1,
                "prioritized_actions": 1,
            },
            "missing_data": [{"column": "income", "missing_rate": 0.2}],
            "outliers": [{"column": "income", "outlier_rate": 0.1}],
            "normality_tests": [{"column": "income", "pvalue": 0.001}],
            "prioritized_actions": [
                {
                    "priority": "high",
                    "column": "income",
                    "action": "fit_imputation_inside_training_fold",
                    "evidence": "missing rate=20%",
                }
            ],
        },
    }
    InteractiveHTMLReport().generate(results, pd.DataFrame({"income": range(10)}), str(output))
    content = output.read_text(encoding="utf-8")
    assert "Diagnóstico e sugestões de pré-processamento" in content
    assert "fit_imputation_inside_training_fold" in content
