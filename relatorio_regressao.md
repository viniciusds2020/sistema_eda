# 📊 Relatório de Análise Exploratória

**Dataset:** Predição de Renda
**Gerado em:** 2026-01-30 17:59:59
**Gerado por:** SmartEDA Python

---


## ℹ Sumário Executivo


<div style="display:flex;flex-wrap:wrap;gap:10px;margin:20px 0;">

<div style="display:inline-block;background:#f8f9fa;border-radius:8px;padding:15px;margin:5px;min-width:150px;text-align:center;">
    <div style="font-size:24px;font-weight:bold;color:#2c3e50;">1,000</div>
    <div style="font-size:14px;color:#7f8c8d;">Linhas</div>
    
</div>


<div style="display:inline-block;background:#f8f9fa;border-radius:8px;padding:15px;margin:5px;min-width:150px;text-align:center;">
    <div style="font-size:24px;font-weight:bold;color:#2c3e50;">14</div>
    <div style="font-size:14px;color:#7f8c8d;">Colunas</div>
    
</div>


<div style="display:inline-block;background:#f8f9fa;border-radius:8px;padding:15px;margin:5px;min-width:150px;text-align:center;">
    <div style="font-size:24px;font-weight:bold;color:#2c3e50;">2</div>
    <div style="font-size:14px;color:#7f8c8d;">Numéricas</div>
    
</div>


<div style="display:inline-block;background:#f8f9fa;border-radius:8px;padding:15px;margin:5px;min-width:150px;text-align:center;">
    <div style="font-size:24px;font-weight:bold;color:#2c3e50;">3</div>
    <div style="font-size:14px;color:#7f8c8d;">Categóricas</div>
    
</div>


<div style="display:inline-block;background:#f8f9fa;border-radius:8px;padding:15px;margin:5px;min-width:150px;text-align:center;">
    <div style="font-size:24px;font-weight:bold;color:#2c3e50;">1</div>
    <div style="font-size:14px;color:#7f8c8d;">Temporais</div>
    
</div>


<div style="display:inline-block;background:#f8f9fa;border-radius:8px;padding:15px;margin:5px;min-width:150px;text-align:center;">
    <div style="font-size:24px;font-weight:bold;color:#2c3e50;">0.7%</div>
    <div style="font-size:14px;color:#7f8c8d;">Ausentes</div>
    
</div>

</div>



## 📋 Visão Geral dos Dados

### Estrutura do Dataset

| Coluna | Tipo Original | Tipo Inferido | Únicos | Ausentes |
| --- | --- | --- | --- | --- |
| id_cliente | int64 | <span style="background-color:#95a5a6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">ID</span> | 1000 | 0.0% |
| idade | int64 | <span style="background-color:#3498db;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Numérico</span> | 51 | 0.0% |
| genero | str | <span style="background-color:#1abc9c;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Binário</span> | 2 | 0.0% |
| estado_civil | str | <span style="background-color:#9b59b6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Categórico</span> | 4 | 0.0% |
| escolaridade | str | <span style="background-color:#9b59b6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Categórico</span> | 4 | 2.3% |
| regiao | str | <span style="background-color:#9b59b6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Categórico</span> | 5 | 0.0% |
| renda_mensal | float64 | <span style="background-color:#95a5a6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">ID</span> | 953 | 4.7% |
| score_credito | float64 | <span style="background-color:#3498db;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Numérico</span> | 355 | 2.1% |
| tempo_emprego_anos | float64 | <span style="background-color:#95a5a6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">ID</span> | 1000 | 0.0% |
| tipo_cliente | str | <span style="background-color:#9b59b6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Categórico</span> | 4 | 0.0% |
| possui_cartao | int64 | <span style="background-color:#1abc9c;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Binário</span> | 2 | 0.0% |
| valor_ultima_compra | float64 | <span style="background-color:#95a5a6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">ID</span> | 1000 | 0.0% |
| data_cadastro | datetime64[us] | <span style="background-color:#e67e22;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Temporal</span> | 728 | 0.0% |
| inadimplente | int64 | <span style="background-color:#1abc9c;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Binário</span> | 2 | 0.0% |

## 🏷️ Inferência de Tipos

### Distribuição por Tipo

- **id**: 4 colunas
- **numeric_discrete**: 1 colunas
- **binary**: 3 colunas
- **unknown**: 4 colunas
- **numeric_continuous**: 1 colunas
- **datetime**: 1 colunas


## 🔢 Análise de Variáveis Numéricas

### Estatísticas Descritivas

| Variável | N | Média | Mediana | Desvio | Mín | Máx | Assimetria | Outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| idade | 1000 | 35.10 | 35.00 | 11.08 | 18.00 | 80.00 | 0.40 | 7 |
| score_credito | 979 | 649.54 | 650.00 | 96.20 | 348.00 | 850.00 | -0.10 | 4 |

### Percentis

| Variável | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| idade | 18.00 | 18.00 | 27.00 | 35.00 | 42.00 | 55.00 | 62.02 |
| score_credito | 425.78 | 493.00 | 585.00 | 650.00 | 716.00 | 812.20 | 850.00 |

### Distribuições

![idade](relatorio_regressao_plots/numeric_idade.png)

![score_credito](relatorio_regressao_plots/numeric_score_credito.png)



## 🏷️ Análise de Variáveis Categóricas

### Resumo

| Variável | N | Únicos | Moda | Moda% | Entropia | Raros |
| --- | --- | --- | --- | --- | --- | --- |
| genero | 1000 | 2 | F | 50.5% | 1.00 | 0 |
| possui_cartao | 1000 | 2 | 1 | 70.6% | 0.87 | 0 |
| inadimplente | 1000 | 2 | 0 | 84.9% | 0.61 | 0 |

### Top Categorias por Variável


#### genero

| Categoria | Contagem | Percentual |
| --- | --- | --- |
| F | 505 | 50.5% |
| M | 495 | 49.5% |

#### possui_cartao

| Categoria | Contagem | Percentual |
| --- | --- | --- |
| 1 | 706 | 70.6% |
| 0 | 294 | 29.4% |

#### inadimplente

| Categoria | Contagem | Percentual |
| --- | --- | --- |
| 0 | 849 | 84.9% |
| 1 | 151 | 15.1% |


### Distribuições

![genero](relatorio_regressao_plots/categorical_genero.png)

![possui_cartao](relatorio_regressao_plots/categorical_possui_cartao.png)

![inadimplente](relatorio_regressao_plots/categorical_inadimplente.png)



## 📅 Análise de Variáveis Temporais

### Resumo

| Variável | N | Início | Fim | Range | Tem Hora | Gaps |
| --- | --- | --- | --- | --- | --- | --- |
| data_cadastro | 1000 | 2020-01-05 | 2024-02-04 | 4.1 anos | Não | 4 |

## 🔗 Análise de Correlações



### Matriz de Correlação

![Correlação](relatorio_regressao_plots/correlation_matrix.png)



## 🎯 Análise com Variável Target

**Target:** renda_mensal
**Tipo:** regression

### Ranking de Features por Importância

| # | Feature | Tipo | Métrica | Valor | Interpretação |
| --- | --- | --- | --- | --- | --- |
| 1 | idade | numeric | Correlação Pearson | -0.0112 | Correlação não significativa (negativa) |
| 2 | inadimplente | categorical | Eta-squared | 0.0059 | Efeito pequeno |
| 3 | score_credito | numeric | Correlação Pearson | -0.0053 | Correlação não significativa (negativa) |
| 4 | genero | categorical | Eta-squared | 0.0013 | Sem diferença significativa entre grupos |
| 5 | possui_cartao | categorical | Eta-squared | 0.0002 | Sem diferença significativa entre grupos |

## ⭐ Importância de Variáveis

### Ranking Consolidado

| # | Feature | Tipo | Mutual Info | Score |
| --- | --- | --- | --- | --- |
| 1 | idade | numeric | - | 0.6000 |
| 2 | inadimplente | categorical | 0.0368 | 0.6000 |
| 3 | score_credito | numeric | 0.0017 | 0.4000 |
| 4 | possui_cartao | categorical | 0.0127 | 0.1732 |
| 5 | genero | categorical | 0.0030 | 0.0000 |




---

*Relatório gerado automaticamente pelo SmartEDA Python em 2026-01-30 18:00:01*
