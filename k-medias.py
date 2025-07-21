import numpy as np
import random

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

def k_medias(dados, k, max_iter=10): # depois veja quantas interações seria bom para o trabalho!!!
    centroides = random.sample(dados, k)

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

    return rotulos, centroides