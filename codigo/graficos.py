import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# ---------------------------------------------------
# Funções auxiliares de visualização
# ---------------------------------------------------

def plot_ari_por_k():
    df = pd.read_excel("resultados_finais.ods")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='k', y='AR_score', hue='algoritmo', style='dataset', markers=True, dashes=False)
    plt.title("Índice Rand Ajustado (ARI) vs Número de Clusters (k)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("ARI")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("grafico_ari_por_k.png")
    plt.show()

def plot_clusters_dispersao(dataset, k, algoritmo):
    caminho_dados = f'../datasets/{dataset}.txt'
    dados_df = pd.read_csv(caminho_dados, sep=r'\s+', header=None, names=['id', 'x', 'y'])

    caminho_rotulos = f'../particoes_geradas/{algoritmo}/particao_{dataset}_k{k}.txt'
    rotulos_df = pd.read_csv(caminho_rotulos, sep=r'\s+', header=None, names=['id', 'cluster'])

    dados_df['cluster'] = rotulos_df['cluster']

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=dados_df, x='x', y='y', hue='cluster', palette='Set1', s=60)
    plt.title(f'{algoritmo} – {dataset} (k={k})')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(f'clusters_{dataset}_{algoritmo}_k{k}.png')
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
    plot_ari_por_k()

    df = pd.read_excel("resultados_finais.ods")
    agrupados = df.groupby(['dataset', 'algoritmo'])

    for (dataset, algoritmo), grupo in agrupados:
        k_melhor = grupo.loc[grupo['AR_score'].idxmax()]['k']

        # Primeiro, dendrograma (se aplicável)
        if algoritmo == 'complete-link':
            plot_dendrogram_complete(dataset)
        elif algoritmo == 'single-link':
            plot_dendrogram_single(dataset)

        # Depois, gráfico de dispersão com melhor k
        plot_clusters_dispersao(dataset, int(k_melhor), algoritmo)

if __name__ == "__main__":
    # from main import executar_algoritmos  # ajuste conforme o nome do seu script
    # executar_algoritmos()
    executar_visualizacoes()