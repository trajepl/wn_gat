from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.datasets import Planetoid


def cluster_scatter(x: np.array, y: np.array, pic_title: str,
                    pos: int, pca_n_components: int = 2) -> None:
    pca = PCA(n_components=2)
    x_pca = pca.fit(x).transform(x)

    plt.subplot(pos)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
    plt.title(pic_title)


def tsne_scatter(x: np.array, y: np.array, pic_title: str,
                 pos: int, tsne_components: int = 2) -> None:
    tsne = TSNE(n_components=2, random_state=2019, init='pca')
    x_tsne = tsne.fit_transform(x)

    plt.subplot(pos)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y)
    plt.title(pic_title)


def pick_candidate_class(x: torch.Tensor, y: torch.Tensor,
                         candidate_class: List, cuda_device: bool = True):
    rls = x[y == candidate_class[0]]
    for item in candidate_class[1:]:
        rls = torch.cat((rls, x[y == item]), dim=0)

    if cuda_device:
        rls = rls.cpu()
    return rls.numpy()


if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    emb_data = torch.load('wn_gat.emb')

    candidate_class = [0, 1, 2, 3]
    random_state = 2019
    # y
    # X = pick_candidate_class(data.x, data.y, candidate_class)
    y = pick_candidate_class(data.y, data.y, candidate_class)

    # y_pred
    x = pick_candidate_class(emb_data, data.y, candidate_class)
    y_pred = KMeans(n_clusters=len(candidate_class),
                    random_state=random_state).fit_predict(x)

    plt.figure(figsize=(12, 12))
    cluster_scatter(x, y_pred, 'pca clustering on y_pred', 221)
    cluster_scatter(x, y, 'pca clustering on y', 222)
    tsne_scatter(x, y_pred, 'tsne clustering on y_pred', 223)
    tsne_scatter(x, y, 'tsne clustering on y', 224)
    plt.show()
