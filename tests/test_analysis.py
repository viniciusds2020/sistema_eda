import pandas as pd

from smarteda import Config, SmartEDA


def test_analysis_returns_overview_and_statistical_sections():
    df = pd.DataFrame(
        {
            "idade": [20, 25, 30, 35, 40, 45],
            "renda": [1500, 2200, 2800, 3500, 4100, 5200],
            "segmento": ["A", "A", "B", "B", "C", "C"],
            "inadimplente": [0, 0, 0, 1, 1, 1],
        }
    )

    eda = SmartEDA(
        df,
        target="inadimplente",
        config=Config(include_plots=False, correlation_threshold=0.0),
        dataset_name="Teste",
    )
    results = eda.analyze()

    assert results["overview"]["n_rows"] == 6
    assert results["overview"]["n_cols"] == 4
    assert "numeric_stats" in results
    assert "correlations" in results
    assert "target_analysis" in results
    assert "importance" in results
