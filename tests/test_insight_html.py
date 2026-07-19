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
