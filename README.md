# SmartEDA

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.3.0-blue)](#)
[![DataFrames](https://img.shields.io/badge/DataFrames-pandas_%7C_Polars_%7C_DuckDB-0A7)](#escala-e-materialização)
[![License](https://img.shields.io/badge/license-MIT-green)](#licença)

Biblioteca Python para análise exploratória, data profiling e monitoramento de drift. Combina estatística descritiva, diagnóstico de qualidade, testes com correção para múltiplas comparações, drift condicionado ao target e acompanhamento longitudinal.

## Capacidades

- análise numérica, categórica, temporal e supervisionada;
- Pearson, Cramér's V e Eta-squared;
- constantes, possíveis IDs e alertas de target leakage;
- PSI, Jensen-Shannon, missing drift e categorias inéditas;
- Kolmogorov–Smirnov e qui-quadrado;
- correções Benjamini–Hochberg FDR e Bonferroni;
- drift por classe ou faixa do target;
- várias janelas comparadas contra uma referência;
- pandas, Polars LazyFrame e DuckDB Relation;
- materialização limitada antes da conversão;
- relatórios Markdown e HTML interativos.

## Instalação

```bash
git clone https://github.com/viniciusds2020/sistema_eda.git
cd sistema_eda
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate # Windows
pip install -e ".[all]"
```

## Fluxo completo

```python
from smarteda import Config, SmartEDA

eda = SmartEDA(
    train_df,
    target="inadimplente",
    dataset_name="Risco de crédito",
    config=Config(
        sample_size=200_000,
        statistical_alpha=0.05,
        pvalue_correction="fdr_bh",
        min_segment_size=100,
        include_plots=False,
    ),
)

results = eda.analyze()
overall = eda.profile_train_test(test_df)
tests = eda.run_distribution_tests(test_df)
conditioned = eda.profile_target_conditioned(test_df)

history = eda.monitor_windows(
    {
        "2026-01": janeiro_df,
        "2026-02": fevereiro_df,
        "2026-03": marco_df,
    }
)

eda.generate_html_report("reports/monitoring.html")
```

## Testes estatísticos e múltiplas comparações

```python
tests = eda.run_distribution_tests(
    test_df,
    correction="fdr_bh",  # ou bonferroni
    alpha=0.05,
)
```

| Tipo de feature | Teste | Tamanho de efeito |
|---|---|---|
| Numérica | KS de duas amostras | estatística KS |
| Categórica | qui-quadrado de homogeneidade | Cramér's V |

O p-valor ajustado controla o volume de falsos positivos produzido ao testar muitas colunas. Significância deve ser interpretada junto com tamanho de efeito, drift operacional e impacto no modelo.

## Drift condicionado ao target

```python
conditioned = eda.profile_target_conditioned(
    test_df,
    target_bins=5,
    min_samples=100,
)
```

- targets discretos são segmentados por classe;
- targets contínuos usam quantis calculados no treino;
- segmentos pequenos são marcados como insuficientes;
- PSI e Jensen-Shannon são recalculados dentro de cada segmento.

Isso permite distinguir mudança global de mudança concentrada em uma classe, faixa de risco ou faixa de valor.

## Escala e materialização

```python
config = Config(sample_size=200_000)
eda = SmartEDA(polars_lazyframe, config=config)
```

O `sample_size` funciona como limite de materialização:

| Fonte | Estratégia |
|---|---|
| pandas | amostragem aleatória reprodutível |
| Polars DataFrame | `sample()` antes da conversão |
| Polars LazyFrame | `limit()` antes de `collect()` |
| DuckDB Relation | `limit()` antes de `.df()` |

Para fontes lazy, o limite é enviado ao motor antes da coleta. A análise continua usando pandas internamente, mas a fonte completa não precisa ser carregada em memória.

## Monitoramento longitudinal

```python
history = eda.monitor_windows(
    {
        "baseline+1": window_1,
        "baseline+2": window_2,
        "baseline+3": window_3,
    }
)
```

Cada janela é comparada contra a mesma referência. O resultado contém:

- resumo por janela;
- histórico por feature;
- nível de drift;
- mudanças de missing e schema;
- série temporal exibida no relatório HTML.

Use definições de janela estáveis — por dia, semana, mês ou lote — para evitar que mudanças de granularidade sejam confundidas com drift.

## Resultados estruturados

```python
results["quality_diagnostics"]
results["train_test_profile"]
results["statistical_drift_tests"]
results["target_conditioned_drift"]
results["longitudinal_monitoring"]
```

## Relatório HTML

O relatório reúne cartões de qualidade, testes corrigidos, segmentos do target, tabelas de janelas e gráficos interativos de drift geral, condicional e longitudinal.

```python
eda.generate_html_report("reports/monitoring.html")
```

## Considerações estatísticas

- p-valor pequeno não implica efeito relevante;
- testes ficam muito sensíveis em amostras grandes;
- correção FDR é menos conservadora que Bonferroni;
- condicionamento no target é diagnóstico e não substitui métricas do modelo;
- PSI e Jensen-Shannon são indicadores operacionais, não testes causais;
- limites e tamanhos mínimos devem ser calibrados por domínio.

## Desenvolvimento

```bash
pip install -e ".[dev,all]"
ruff check smarteda tests
pytest --cov=smarteda --cov-report=term-missing
```

## Licença

MIT.

## Autor

Desenvolvido por [Vinicius de Sousa](https://github.com/viniciusds2020).
