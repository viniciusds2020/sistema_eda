from pathlib import Path

import pandas as pd

from smarteda import Config, SmartEDA


def test_interactive_html_report_is_created(tmp_path: Path):
    df = pd.DataFrame(
        {
            "id": range(20),
            "value": range(20),
            "group": ["A"] * 10 + ["B"] * 10,
            "target": [0, 1] * 10,
        }
    )
    eda = SmartEDA(df, target="target", config=Config(include_plots=False))
    eda.analyze()
    eda.profile_train_test(df.copy())

    output = tmp_path / "report.html"
    eda.generate_html_report(str(output))

    content = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "SmartEDA" in content
    assert "plotly" in content.lower()
