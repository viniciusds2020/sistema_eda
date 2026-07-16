# SmartEDA

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.2.0-blue)](#)
[![DataFrames](https://img.shields.io/badge/DataFrames-pandas_%7C_Polars_%7C_DuckDB-0A7)](#fontes-de-dados)
[![License](https://img.shields.io/badge/license-MIT-green)](#licença)

Biblioteca Python para análise exploratória e data profiling. O SmartEDA combina estatística descritiva, associações entre tipos mistos, diagnóstico de qualidade, alertas de leakage, comparação treino–teste e relatórios HTML interativos.

## Principais capacidades

- inferência de variáveis numéricas, categóricas, temporais, binárias e identificadores;
- estatísticas descritivas, percentis, assimetria, outliers e missing values;
- Pearson, Cramér's V e Eta-squared para associações entre tipos diferentes;
- análise de target para classificação e regressão;
- ranking de importância com sinais estatísticos e Random Forest;
- detector de constantes, possíveis IDs e sinais de target leakage;
- PSI para variáveis numéricas e Jensen-Shannon para categóricas;
- taxa de categorias inéditas e mudança de missing entre treino e teste;
- entrada pandas, Polars ou relação DuckDB;
- relatório Markdown e HTML autocontido com Plotly;
- testes, lint e CI em múltiplas versões do Python.

## Instalação

```bash
git clone https://github.com/viniciusds2020/sistema_eda.git
cd sistema_eda
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -e .
```

Para habilitar Polars e DuckDB:

```bash
pip install -e ".[all]"
```

Ambiente de desenvolvimento:

```bash
pip install -e ".[dev,all]"
ruff check .
pytest --cov=smarteda --cov-report=term-missing
```

## Uso completo

```python
import pandas as pd
from smarteda import Config, SmartEDA

train = pd.read_parquet("train.parquet")
test = pd.read_parquet("test.parquet")

eda = SmartEDA(
    train,
    target="inadimplente",
    config=Config(
        include_plots=False,
        sample_size=100_000,
        random_state=42,
    ),
    dataset_name="Risco de crédito",
)

results = eda.analyze()
drift = eda.profile_train_test(test)

eda.generate_report("reports/eda.md")
eda.generate_html_report("reports/eda.html")

print(results["quality_diagnostics"]["summary"])
print(drift.sort_values("drift_score", ascending=False).head(10))
```

## Fontes de dados

### pandas

```python
eda = SmartEDA(pandas_df)
```

### Polars

```python
import polars as pl

eda = SmartEDA(pl.read_parquet("dados.parquet"))
eda_lazy = SmartEDA(pl.scan_parquet("dados.parquet"))
```

O `LazyFrame` é materializado no momento da conversão porque as análises estatísticas utilizam pandas internamente.

### DuckDB

```python
import duckdb

relation = duckdb.sql("SELECT * FROM read_parquet('dados/*.parquet')")
eda = SmartEDA(relation)
```

A compatibilidade é feita por uma camada de adaptação. Isso mantém uma única implementação estatística e uma API consistente.

## Diagnóstico de qualidade e leakage

O resultado de `analyze()` inclui:

```python
diagnostics = results["quality_diagnostics"]

diagnostics["constant_columns"]
diagnostics["possible_ids"]
diagnostics["possible_leakage"]
diagnostics["missing_by_column"]
```

Os alertas de leakage verificam:

- nomes associados a target, resultado ou prediction;
- cópias exatas da variável-alvo;
- correlação numérica quase perfeita com o target.

Essas regras são heurísticas. Um alerta indica necessidade de investigação, não prova de vazamento.

## Profiling treino × teste

```python
profile = eda.profile_train_test(test_df)
```

| Tipo | Indicador | Faixas de triagem |
|---|---|---|
| Numérico | PSI | médio ≥ 0,10; alto ≥ 0,25 |
| Categórico | Jensen-Shannon | médio ≥ 0,10; alto ≥ 0,20 |
| Categórico | Categorias inéditas | taxa observada no teste |
| Todos | Missing delta | teste − treino |
| Schema | Colunas e dtypes | novas, ausentes ou alteradas |

Os limites são referências operacionais e devem ser calibrados conforme sensibilidade do modelo e impacto de negócio.

## Relatório HTML interativo

```python
eda.generate_html_report("reports/eda.html")
```

O arquivo HTML contém:

- cartões de qualidade;
- alertas de IDs, constantes e leakage;
- missing values;
- histogramas;
- correlações;
- distribuições categóricas;
- ranking de drift quando um teste foi comparado.

O Plotly é incorporado ao arquivo para que o relatório possa ser aberto e compartilhado como artefato único.

## API principal

| API | Resultado |
|---|---|
| `SmartEDA(...).analyze()` | perfil estatístico e diagnósticos |
| `profile_train_test(test_df)` | DataFrame de comparação |
| `generate_report(path)` | relatório Markdown |
| `generate_html_report(path)` | relatório HTML interativo |
| `detect_quality_issues(df, target=...)` | diagnósticos isolados |
| `profile_train_test(train, test)` | comparação sem instanciar SmartEDA |
| `to_pandas(frame)` | adaptação de pandas, Polars ou DuckDB |

## Considerações estatísticas

- Correlação e importância não demonstram causalidade.
- PSI e Jensen-Shannon são indicadores de mudança, não explicam a causa.
- Leakage precisa ser validado contra o momento em que cada feature fica disponível.
- Dados de treino e teste devem preservar a ordem temporal quando o problema exigir.
- Amostragem reduz custo, mas pode ocultar categorias raras.
- Variáveis com drift elevado não precisam ser removidas automaticamente; avalie impacto no modelo.

## Limitações e roadmap

- [ ] testes estatísticos com correção para múltiplas comparações;
- [ ] relatório de drift condicionado ao target;
- [ ] suporte a datasets maiores sem materialização completa;
- [ ] monitoramento longitudinal com múltiplas janelas;
- [ ] publicação no PyPI e documentação da API.

## Licença

MIT.

## Autor

Desenvolvido por [Vinicius de Sousa](https://github.com/viniciusds2020).
