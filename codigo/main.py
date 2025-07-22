import pandas as pd
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
import random
import math


# ==============================================================
# Funções de I/O (Entrada e Saída)
# ==============================================================

def carregar_entrada(caminho_input, caminho_clu):
    # Carrega os dados e os rótulos reais dos arquivos de texto
    colunas_dados = ['id', 'x', 'y']
    dados_df = pd.read_csv(caminho_input, sep=r'\s+', header=None, names=colunas_dados)
    ids = dados_df['id'].values
    atributos = dados_df[['x', 'y']].values

    colunas_clusters = ['id', 'cluster']
    clusters_df = pd.read_csv(caminho_clu, sep=r'\s+', header=None, names=colunas_clusters)
    labels_reais = clusters_df['cluster'].values
    
    return ids, atributos, labels_reais

def salvar_saida(caminho_saida, ids, rotulos):
    # Salva uma partição em um arquivo de texto
    saida_df = pd.DataFrame({'id': ids, 'cluster': rotulos})
    saida_df.to_csv(caminho_saida, index=False, header=False, sep=' ')
    print(f"  -> Partição salva em: {os.path.basename(caminho_saida)}")

# ==============================================================
# Implementação K-média
# ==============================================================

# distancia euclidiana
def distancia(p1, p2):
    diff = np.array(p1) - np.array(p2)
    return np.sqrt(np.sum(diff ** 2))

# cálculo das médias
def media_dos_pontos(cluster):
    if not cluster:
        return []
    
    n = len(cluster)
    m = len(cluster[0])

    medias = []
    for j in range(m):
        soma = sum(ponto[j] for ponto in cluster)
        medias.append(soma / n)
    return medias

def indice_menor_valor(lista):
    menor = min(lista)
    return lista.index(menor)

def pontos_do_cluster_i(dados, rotulos, i):
    cluster = []
    for j in range(len(dados)):
        if rotulos[j] == i:
            cluster.append(dados[j])
    return cluster

def k_medias(dados, k, max_iter=10): 

    centroides = random.sample(list(dados), k)

    for _ in range(max_iter):
        rotulos = []
        for ponto in dados:
            distancias = [distancia(ponto, c) for c in centroides]
            rotulos.append(indice_menor_valor(distancias))

        novos_centroides = []
        for i in range(k):
            cluster_i = pontos_do_cluster_i(dados, rotulos, i)
            if cluster_i:
                novos_centroides.append(media_dos_pontos(cluster_i))
            else:
                novos_centroides.append(random.choice(dados))
        centroides = novos_centroides

    return rotulos

# ==============================================================
# Implementação Complete Link
# ==============================================================

# Função auxiliar para encontrar a maior distância entre dois cluster
def distancia_complete_link(cluster1, cluster2, matriz_distancias):
    max_distancia = 0.0
    for i in cluster1:
        for j in cluster2:
            # Consulta a distância na matriz em vez de recalcular
            dist = matriz_distancias[i][j]
            if dist > max_distancia:
                max_distancia = dist
    return max_distancia

#Função auxiliar para converter a lista de clusters em rótulos.
def gerar_rotulos(clusters, num_pontos):

    rotulos = [0] * num_pontos
    for cluster_id, cluster in enumerate(clusters):
        for ponto_idx in cluster:
            rotulos[ponto_idx] = cluster_id
    return rotulos

def complete_link(dados, kMin, kMax):

    num_pontos = len(dados)
    
    # Calcula a matriz de distâncias previamente
    matriz_distancias = squareform(pdist(dados, metric='euclidean')).tolist()
    
    # inicializa os clusters como listas de índices
    clusters = [[i] for i in range(num_pontos)]
    particoes_resultado = {}

    if kMin <= len(clusters) <= kMax:
        particoes_resultado[len(clusters)] = gerar_rotulos(clusters, num_pontos)

    # Loop para fazer a união dos clusters
    while len(clusters) > kMin:
        menor_distancia = math.inf
        par_a_unir = (-1, -1)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Busca a maior distância entre os clusters
                dist = distancia_complete_link(clusters[i], clusters[j], matriz_distancias)
                if dist < menor_distancia:
                    menor_distancia = dist
                    par_a_unir = (i, j)
        
        # Une os dois clusters
        i, j = par_a_unir
        novo_cluster = clusters[i] + clusters[j]
        clusters.pop(max(i, j))
        clusters.pop(min(i, j))
        clusters.append(novo_cluster)
        
        # Salva a partição
        num_clusters_atual = len(clusters)
        if kMin <= num_clusters_atual <= kMax:
            rotulos = gerar_rotulos(clusters, num_pontos)
            particoes_resultado[num_clusters_atual] = rotulos

    return particoes_resultado

# ==============================================================
# Implementação Single Link
# ==============================================================

def single_link(dados, kMin, kMax):
    
    num_pontos = len(dados)
    
    # Calcula a matriz de distâncias previamente
    dist_matrix = squareform(pdist(dados, metric='euclidean'))

    # infinito na diagonal para que um cluster não seja unido com ele mesmo
    np.fill_diagonal(dist_matrix, float('inf'))

    # Inicializando cada ponto como um cluster separado
    clusters = [[i] for i in range(num_pontos)]
    particoes_resultado = {}

    # Salva a partição inicial se estiver no intervalo
    if kMin <= len(clusters) <= kMax:
        particoes_resultado[len(clusters)] = gerar_rotulos(clusters, num_pontos)

    for _ in range(num_pontos - kMin):
        # Para o loop se só sobrar um cluster
        if len(clusters) <= 1: break 

        # Encontra o par mais próximo na matriz de distâncias atual
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        
        # Garante que i < j
        if i > j: i, j = j, i

        # Une os dois clusters mais próximos
        cluster1_indices = clusters[i]
        cluster2_indices = clusters[j]
        cluster_unido = cluster1_indices + cluster2_indices
        
        # Atualizando a matriz de distâncias Para refletir a união
        nova_linha = np.minimum(dist_matrix[i, :], dist_matrix[j, :])
        
        # Remove a linha/coluna do segundo cluster unido 
        dist_matrix = np.delete(dist_matrix, j, axis=0) 
        dist_matrix = np.delete(dist_matrix, j, axis=1) 

        nova_linha = np.delete(nova_linha, j)

        # Substitui a linha/coluna do primeiro cluster unido (i) pela nova linha calculada
        dist_matrix[i, :] = nova_linha
        dist_matrix[:, i] = nova_linha

        # Garante que a diagonal seja infinita
        np.fill_diagonal(dist_matrix, float('inf'))
        
        # Atualiza a lista de clusters
        clusters.pop(j) 
        clusters[i] = cluster_unido
        
        # Salva a partição se o número de clusters atual estiver no intervalo
        num_clusters_atual = len(clusters)
        if kMin <= num_clusters_atual <= kMax:
            rotulos = gerar_rotulos(clusters, num_pontos)
            particoes_resultado[num_clusters_atual] = rotulos

    return particoes_resultado


# ==============================================================
# Função Principal de Execução
# ==============================================================

def executar_algoritmos():
    datasets = {
        'c2ds1-2sp': {'k_range': range(2, 6), 'real_clu': 'c2ds1-2spReal.clu'},
        'c2ds3-2g': {'k_range': range(2, 6), 'real_clu': 'c2ds3-2gReal.clu'},
        'monkey': {'k_range': range(5, 13), 'real_clu': 'monkeyReal1.clu'}
    }
    
    algoritmos = ['k-media', 'single-link', 'complete-link']
    
    resultados_finais = []

    for nome_base, info in datasets.items():
        print(f"\n--- Processando o dataset: {nome_base} ---")
        caminho_dados = f'../datasets/{nome_base}.txt'
        caminho_clu = f'../datasets/{info["real_clu"]}'
        
        ids, atributos, labels_reais = carregar_entrada(caminho_dados, caminho_clu)
        
        for nome_algoritmo in algoritmos:
            print(f"\nExecutando o algoritmo: {nome_algoritmo.upper()}...")
            
            k_min = min(info['k_range'])
            k_max = max(info['k_range'])
            
            particoes_geradas = {}
            
            if nome_algoritmo == 'k-media':
                # K-media precisa ser rodado para cada k individualmente
                for k in info['k_range']:
                    particoes_geradas[k] = k_medias(atributos, k)

            elif nome_algoritmo == 'single-link': 
                # Single-link
                particoes_geradas = single_link(atributos, k_min, k_max)
            
            else:
                # Complete-link
                particoes_geradas = complete_link(atributos, k_min, k_max)
            
            # Salva e avalia cada partição gerada pelo algoritmo
            diretorio_saida = f'../particoes_geradas/{nome_algoritmo}'
            
            for k, rotulos_gerados in particoes_geradas.items():
                nome_arquivo = f'particao_{nome_base}_k{k}.txt'
                caminho_completo = os.path.join(diretorio_saida, nome_arquivo)
                salvar_saida(caminho_completo, ids, rotulos_gerados)
                
                score_ar = adjusted_rand_score(labels_reais, rotulos_gerados)
                print(f"  -> k={k}, Índice Rand Ajustado (AR): {score_ar:.4f}")
                
                resultados_finais.append({
                    'dataset': nome_base,
                    'algoritmo': nome_algoritmo,
                    'k': k,
                    'AR_score': score_ar
                })

    # Salva a tabela consolidada de resultados
    resultados_df = pd.DataFrame(resultados_finais)
    resultados_df.to_excel("resultados_finais.xlsx", index=False)
    print("\n\n--- Tabela de resultados salva em 'resultados_finais.xlsx' ---")
    print(resultados_df)

if __name__ == "__main__":
    executar_algoritmos()