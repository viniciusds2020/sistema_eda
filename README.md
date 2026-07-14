# SmartEDA

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#licença)

Biblioteca Python para automatizar análise exploratória de dados, inferir tipos de variáveis e gerar relatórios reproduzíveis. Inspirada no SmartEDA do ecossistema R.

## Objetivo

A análise exploratória costuma repetir verificações de qualidade, distribuição, associação e relação com a variável-alvo. O SmartEDA organiza esse processo em uma API única e configurável, mantendo acesso aos resultados intermediários.

## Funcionalidades

- inferência de variáveis numéricas, categóricas, temporais e binárias;
- estatísticas descritivas e diagnóstico de valores ausentes;
- detecção de outliers;
- correlação de Pearson, Cramér's V e Eta-squared;
- análise supervisionada para classificação e regressão;
- importância de variáveis com Random Forest;
- relatórios em Markdown com gráficos;
- configuração de limiares e volume de visualizações.

## Instalação

```bash
git clone https://github.com/viniciusds2020/sistema_eda.git
cd sistema_eda

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate # Windows

pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

## Uso rápido

```python
import pandas as pd
from smarteda import SmartEDA

df = pd.read_csv("dados.csv")

eda = SmartEDA(
    df,
    dataset_name="Clientes",
)

eda.analyze()
eda.generate_report("relatorio_eda.md")
```

## Análise com variável-alvo

```python
from smarteda import Config, SmartEDA

config = Config(
    categorical_threshold=15,
    top_n_categories=8,
    include_plots=True,
    correlation_threshold=0.30,
)

eda = SmartEDA(
    df,
    target="inadimplente",
    config=config,
    dataset_name="Risco de crédito",
)

eda.analyze()

print(eda.get_target_summary())
print(eda.get_importance_summary())
eda.generate_report("relatorio_risco.md")
```

## Fluxo da análise

```mermaid
flowchart LR
    D["DataFrame"] --> T["Inferência de tipos"]
    T --> Q["Qualidade e estatísticas"]
    Q --> C["Associações"]
    C --> A["Análise do target"]
    A --> R["Relatório"]
```

As etapas de target e importância são executadas quando uma variável-alvo é informada.

## API principal

| Método | Resultado |
|---|---|
| `analyze()` | executa o pipeline completo |
| `generate_report(path)` | cria o relatório em Markdown |
| `get_numeric_summary()` | resumo de variáveis numéricas |
| `get_categorical_summary()` | resumo de variáveis categóricas |
| `get_correlation_summary()` | associações relevantes |
| `get_target_summary()` | relação com a variável-alvo |
| `get_importance_summary()` | ranking de importância |

## Estrutura

```text
smarteda/
├── core/              # orquestração, configuração e tipos
├── analysis/          # análises estatísticas
├── report/            # geração e estilos
└── utils/             # funções auxiliares
main.py                # exemplos de uso
```

## Princípios do projeto

- oferecer uma primeira leitura consistente do dataset;
- distinguir correlação de diferentes combinações de tipos;
- manter parâmetros explícitos e reproduzíveis;
- produzir artefatos que possam ser revisados e versionados.

## Limitações

- EDA automatizada não substitui conhecimento do domínio;
- importância de Random Forest não representa causalidade;
- inferência de tipos pode exigir ajustes em dados ambíguos;
- grandes volumes podem demandar amostragem;
- decisões sobre outliers e missing values permanecem com o analista.

## Roadmap

- [ ] empacotamento e publicação no PyPI;
- [ ] testes automatizados e CI;
- [ ] relatório HTML;
- [ ] suporte a amostragem e datasets maiores;
- [ ] documentação da API.

## Licença

MIT.

## Autor

Desenvolvido por [Vinicius de Sousa](https://github.com/viniciusds2020).
