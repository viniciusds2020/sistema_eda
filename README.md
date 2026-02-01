# SmartEDA - Análise Exploratória de Dados Inteligente

Biblioteca Python para análise exploratória de dados (EDA) automatizada, inspirada no pacote SmartEDA do R.

## Funcionalidades

- **Inferência automática de tipos**: Detecta automaticamente variáveis numéricas, categóricas, temporais e binárias
- **Análise estatística completa**: Estatísticas descritivas, detecção de outliers, valores ausentes
- **Análise de correlação**: Correlação de Pearson, Cramér's V e Eta-squared para variáveis mistas
- **Análise com variável target**: Suporte para classificação e regressão
- **Importância de variáveis**: Ranking de features usando Random Forest
- **Geração de relatórios**: Relatórios em Markdown com gráficos

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/viniciusds2020/sistema_eda.git
cd sistema_eda

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

## Uso Rápido

```python
import pandas as pd
from smarteda import SmartEDA, Config

# Carregar dados
df = pd.read_csv("seus_dados.csv")

# Análise simples
eda = SmartEDA(df, dataset_name="Meu Dataset")
eda.analyze()
eda.generate_report("relatorio.md")

# Análise com variável target (classificação)
eda = SmartEDA(df, target="coluna_target")
eda.analyze()
eda.generate_report("relatorio_classificacao.md")
```

## Exemplos

### Análise Básica

```python
from smarteda import SmartEDA

eda = SmartEDA(df, dataset_name="Clientes")
eda.analyze()

# Visualizar resumos
print(eda.get_numeric_summary())
print(eda.get_categorical_summary())
print(eda.get_correlation_summary())
```

### Análise com Target e Configuração Personalizada

```python
from smarteda import SmartEDA, Config

config = Config(
    categorical_threshold=15,      # Máximo de categorias únicas
    top_n_categories=8,            # Top N categorias nos gráficos
    include_plots=True,            # Gerar gráficos
    correlation_threshold=0.3      # Limiar para correlações significativas
)

eda = SmartEDA(
    df,
    target="inadimplente",
    config=config,
    dataset_name="Análise de Risco"
)
eda.analyze()

# Ranking de importância de variáveis
print(eda.get_importance_summary())
```

## Estrutura do Projeto

```
sistema_eda/
├── smarteda/
│   ├── __init__.py
│   ├── core/
│   │   ├── analyzer.py      # Classe principal SmartEDA
│   │   ├── config.py        # Configurações
│   │   └── type_inference.py # Inferência de tipos
│   ├── analysis/
│   │   ├── numeric.py       # Análise numérica
│   │   ├── categorical.py   # Análise categórica
│   │   ├── temporal.py      # Análise temporal
│   │   ├── correlation.py   # Análise de correlação
│   │   ├── target.py        # Análise com target
│   │   └── importance.py    # Importância de variáveis
│   ├── report/
│   │   ├── generator.py     # Gerador de relatórios
│   │   └── styles.py        # Estilos do relatório
│   └── utils/
│       └── helpers.py       # Funções auxiliares
├── main.py                  # Exemplos de uso
└── README.md
```

## Dependências

- Python 3.8+
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn

## Métodos Principais

| Método | Descrição |
|--------|-----------|
| `analyze()` | Executa análise exploratória completa |
| `generate_report(path)` | Gera relatório em Markdown |
| `get_numeric_summary()` | Retorna resumo das variáveis numéricas |
| `get_categorical_summary()` | Retorna resumo das variáveis categóricas |
| `get_correlation_summary()` | Retorna correlações significativas |
| `get_target_summary()` | Retorna análise com target |
| `get_importance_summary()` | Retorna ranking de importância |

## Licença

MIT License
