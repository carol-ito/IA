import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# ---------------------------------------------------
# Funções auxiliares de visualização
# ---------------------------------------------------

def plot_ari_por_k_por_dataset():
    df = pd.read_excel("resultados_finais.xlsx")

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=subset, x='k', y='AR_score', hue='algoritmo', marker='o')
        plt.title(f"ARI vs k – Dataset: {dataset}")
        plt.xlabel("Número de Clusters (k)")
        plt.ylabel("ARI")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"grafico_ari_por_k_{dataset}.png")
        plt.show()

def plot_comparacao_clusters(dataset, k, algoritmo):
    # --- Carrega dados base (x, y) ---
    caminho_dados = f'../datasets/{dataset}.txt'
    dados_df = pd.read_csv(caminho_dados, sep=r'\s+', header=None, names=['id', 'x', 'y'])

    # --- Clusters gerados pelo algoritmo ---
    caminho_rotulos = f'../particoes_geradas/{algoritmo}/particao_{dataset}_k{k}.txt'
    rotulos_df = pd.read_csv(caminho_rotulos, sep=r'\s+', header=None, names=['id', 'cluster'])
    dados_cluster = pd.merge(dados_df, rotulos_df, on='id')

    # --- Tenta carregar rótulos reais com id + label ---
    if dataset == 'monkey':
        caminho_reais = f'../datasets/{dataset}Real1.clu'
    else:
        caminho_reais = f'../datasets/{dataset}Real.clu'
    
    possui_reais = os.path.exists(caminho_reais)

    if possui_reais:
        rotulos_reais = pd.read_csv(caminho_reais, sep=r'\s+', header=None, names=['id', 'label'])
        dados_reais = pd.merge(dados_df, rotulos_reais, on='id')

    # --- Plot side by side ---
    fig, axes = plt.subplots(1, 2 if possui_reais else 1, figsize=(12 if possui_reais else 6, 6))
    if not possui_reais:
        axes = [axes]

    # Clusters gerados
    sns.scatterplot(data=dados_cluster, x='x', y='y', hue='cluster', palette='Set1', s=60, ax=axes[0])
    axes[0].set_title(f'{algoritmo} – {dataset} (k={k})')
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Rótulos reais (se houver)
    if possui_reais:
        sns.scatterplot(data=dados_reais, x='x', y='y', hue='label', palette='Dark2', s=60, ax=axes[1])
        axes[1].set_title(f'Rótulos Reais – {dataset}')
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.savefig(f'comparacao_{dataset}_{algoritmo}_k{k}.png')
    plt.show()

def plot_dendrogram_complete(dataset):
    caminho_dados = f'../datasets/{dataset}.txt'
    dados_df = pd.read_csv(caminho_dados, sep=r'\s+', header=None, names=['id', 'x', 'y'])
    dados = dados_df[['x', 'y']].values

    Z = linkage(dados, method='complete', metric='euclidean')

    plt.figure(figsize=(10, 6))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8, no_labels=True)
    plt.title(f'Dendrograma – Complete Link ({dataset})')
    plt.xlabel("Pontos")
    plt.ylabel("Distância Euclidiana")
    plt.tight_layout()
    plt.savefig(f'dendrograma_complete_{dataset}.png')
    plt.show()

def plot_dendrogram_single(dataset):
    caminho_dados = f'../datasets/{dataset}.txt'
    dados_df = pd.read_csv(caminho_dados, sep=r'\s+', header=None, names=['id', 'x', 'y'])
    dados = dados_df[['x', 'y']].values

    Z = linkage(dados, method='single', metric='euclidean')

    plt.figure(figsize=(10, 6))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8, no_labels=True)
    plt.title(f'Dendrograma – Single Link ({dataset})')
    plt.xlabel("Pontos")
    plt.ylabel("Distância Euclidiana")
    plt.tight_layout()
    plt.savefig(f'dendrograma_single_{dataset}.png')
    plt.show()

# ---------------------------------------------------
# Executa tudo
# ---------------------------------------------------

def executar_visualizacoes():
    plot_ari_por_k_por_dataset()

    df = pd.read_excel("resultados_finais.xlsx")
    agrupados = df.groupby(['dataset', 'algoritmo'])

    for (dataset, algoritmo), grupo in agrupados:
        k_melhor = grupo.loc[grupo['AR_score'].idxmax()]['k']

        # Primeiro, dendrograma (se aplicável)
        if algoritmo == 'complete-link':
            plot_dendrogram_complete(dataset)
        elif algoritmo == 'single-link':
            plot_dendrogram_single(dataset)

        plot_comparacao_clusters(dataset, int(k_melhor), algoritmo)

if __name__ == "__main__":
    executar_visualizacoes()