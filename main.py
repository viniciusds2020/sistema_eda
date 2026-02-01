"""
SmartEDA - Exemplo de Uso
=========================

Este script demonstra como utilizar o pacote SmartEDA para
análise exploratória de dados automatizada.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Adicionar o diretório ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smarteda import SmartEDA, Config


def create_sample_dataset(n_rows: int = 1000) -> pd.DataFrame:
    """Cria dataset de exemplo para demonstração."""
    np.random.seed(42)

    # Variáveis numéricas
    idade = np.random.normal(35, 12, n_rows).clip(18, 80).astype(int)
    renda = np.random.lognormal(10, 0.8, n_rows)
    score_credito = np.random.normal(650, 100, n_rows).clip(300, 850).astype(int)
    tempo_emprego = np.random.exponential(5, n_rows).clip(0, 40)
    valor_compra = np.random.lognormal(5, 1.5, n_rows)

    # Variáveis categóricas
    genero = np.random.choice(['M', 'F'], n_rows)
    estado_civil = np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], n_rows, p=[0.3, 0.5, 0.15, 0.05])
    escolaridade = np.random.choice(['Fundamental', 'Médio', 'Superior', 'Pós-graduação'], n_rows, p=[0.1, 0.35, 0.40, 0.15])
    regiao = np.random.choice(['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'], n_rows, p=[0.08, 0.27, 0.07, 0.42, 0.16])
    tipo_cliente = np.random.choice(['Bronze', 'Prata', 'Ouro', 'Platina'], n_rows, p=[0.5, 0.3, 0.15, 0.05])

    # Variável temporal
    base_date = datetime(2020, 1, 1)
    dias_aleatorios = np.random.randint(0, 1500, n_rows)
    data_cadastro = [base_date + timedelta(days=int(d)) for d in dias_aleatorios]

    # Variável binária
    possui_cartao = np.random.choice([0, 1], n_rows, p=[0.3, 0.7])

    # Target (classificação binária - risco de inadimplência)
    # Baseado em algumas features
    prob_inadimplencia = (
        0.1 +
        0.2 * (score_credito < 500).astype(float) +
        0.15 * (tempo_emprego < 1).astype(float) +
        0.1 * (idade < 25).astype(float) +
        np.random.uniform(-0.1, 0.1, n_rows)
    ).clip(0, 1)
    inadimplente = (np.random.random(n_rows) < prob_inadimplencia).astype(int)

    # Criar DataFrame
    df = pd.DataFrame({
        'id_cliente': range(1, n_rows + 1),
        'idade': idade,
        'genero': genero,
        'estado_civil': estado_civil,
        'escolaridade': escolaridade,
        'regiao': regiao,
        'renda_mensal': renda,
        'score_credito': score_credito,
        'tempo_emprego_anos': tempo_emprego,
        'tipo_cliente': tipo_cliente,
        'possui_cartao': possui_cartao,
        'valor_ultima_compra': valor_compra,
        'data_cadastro': data_cadastro,
        'inadimplente': inadimplente
    })

    # Adicionar alguns valores ausentes
    mask_renda = np.random.random(n_rows) < 0.05
    df.loc[mask_renda, 'renda_mensal'] = np.nan

    mask_score = np.random.random(n_rows) < 0.03
    df.loc[mask_score, 'score_credito'] = np.nan

    mask_escolaridade = np.random.random(n_rows) < 0.02
    df.loc[mask_escolaridade, 'escolaridade'] = np.nan

    return df


def exemplo_analise_simples():
    """Exemplo de análise simples sem variável target."""
    print("\n" + "="*60)
    print("EXEMPLO 1: Análise Simples (sem target)")
    print("="*60 + "\n")

    # Criar dataset
    df = create_sample_dataset(1000)
    print(f"Dataset criado: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # Criar instância do SmartEDA
    eda = SmartEDA(df, dataset_name="Clientes - Análise Geral")

    # Executar análise
    eda.analyze()

    # Gerar relatório
    eda.generate_report("relatorio_simples.md")

    # Visualizar resumos
    print("\n📊 Resumo Numérico:")
    print(eda.get_numeric_summary().to_string())

    print("\n🏷️ Resumo Categórico:")
    print(eda.get_categorical_summary().to_string())

    print("\n🔗 Correlações Significativas:")
    corr_summary = eda.get_correlation_summary()
    if not corr_summary.empty:
        print(corr_summary.head(10).to_string())


def exemplo_analise_com_target():
    """Exemplo de análise com variável target."""
    print("\n" + "="*60)
    print("EXEMPLO 2: Análise com Target (Classificação)")
    print("="*60 + "\n")

    # Criar dataset
    df = create_sample_dataset(1000)
    print(f"Dataset criado: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # Configuração personalizada
    config = Config(
        categorical_threshold=15,
        top_n_categories=8,
        include_plots=True,
        correlation_threshold=0.3
    )

    # Criar instância com target
    eda = SmartEDA(
        df,
        target="inadimplente",
        config=config,
        dataset_name="Análise de Risco de Crédito"
    )

    # Executar análise
    eda.analyze()

    # Gerar relatório
    eda.generate_report("relatorio_com_target.md")

    # Visualizar resumos
    print("\n🎯 Análise com Target:")
    target_summary = eda.get_target_summary()
    if not target_summary.empty:
        print(target_summary.to_string())

    print("\n⭐ Importância de Variáveis:")
    importance_summary = eda.get_importance_summary()
    if not importance_summary.empty:
        print(importance_summary.head(10).to_string())


def exemplo_analise_regressao():
    """Exemplo de análise com target de regressão."""
    print("\n" + "="*60)
    print("EXEMPLO 3: Análise com Target (Regressão)")
    print("="*60 + "\n")

    # Criar dataset
    df = create_sample_dataset(1000)
    print(f"Dataset criado: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # Usar renda_mensal como target (regressão)
    eda = SmartEDA(
        df,
        target="renda_mensal",
        dataset_name="Predição de Renda"
    )

    # Executar análise
    eda.analyze()

    # Gerar relatório
    eda.generate_report("relatorio_regressao.md")

    print("\n🎯 Top Features para Predição de Renda:")
    importance_summary = eda.get_importance_summary()
    if not importance_summary.empty:
        print(importance_summary.head(10).to_string())


def main():
    """Função principal - executa todos os exemplos."""
    print("\n" + "#"*60)
    print("#" + " SmartEDA - Demonstração Completa ".center(58) + "#")
    print("#"*60)

    # Verificar dependências
    try:
        import scipy
        import sklearn
        print("\n[OK] Todas as dependencias estao instaladas")
    except ImportError as e:
        print(f"\n[AVISO] Dependencia faltando: {e}")
        print("   Instale com: pip install scipy scikit-learn matplotlib seaborn")
        return

    # Executar exemplos
    exemplo_analise_simples()
    exemplo_analise_com_target()
    exemplo_analise_regressao()

    print("\n" + "="*60)
    print("✅ Todos os exemplos foram executados!")
    print("📄 Relatórios gerados:")
    print("   - relatorio_simples.md")
    print("   - relatorio_com_target.md")
    print("   - relatorio_regressao.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
