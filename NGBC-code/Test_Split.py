import os
import time
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler
from NaturalBall.NaturalBallClustering_synthetic import nb_plot, form_nb_cluster_by_neighbor
from Naturalneighbor_Split import *
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.manifold import TSNE


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


# 获取文件路径
def through(lujing, key_word):
    file_list = []
    count = 0
    for root, dirs, files in os.walk(lujing):
        for file in files:
            name = os.path.join(root, file)
            if key_word in name:
                count += 1
                file_list.append(name)
    return file_list


if __name__ == '__main__':

    pathList = through(r".\syndataset", ".csv")

    # 跑某些范围
    pathList = pathList[0:len(pathList)]
    for i, dataPath in enumerate(pathList):
        beginTime = time.time()
        # 只跑某个文件时使用，跑所有数据将if注掉
        if "D2.csv" not in dataPath:
            continue

        dataName = dataPath.split("\\")[-1]
        print(f"加载{dataName}....")

        df = pd.read_csv(dataPath, header=None)  # 加载数据集

        # 不取标签那一列，否则对距离矩阵的计算有影响，导致影响球的形成
        data = df.values[:, :2]
        # data = df.values[:, 1:]

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        # tsne = TSNE(n_components=2, init='pca')
        # data = tsne.fit_transform(data)

        N = len(data)
        D = pdist(data)
        A = squareform(D)

        # 自然球--->自然簇
        NNtool = NNSearch(A)
        t, nn, rnn, dis_index = NNtool.natural_search()
        nbGroup = NNtool.get_nb_group(nn, rnn, data)
        nbCluster = form_nb_cluster_by_neighbor(data, dis_index, nbGroup)
        endTime = time.time()
        print(f"{dataName}耗时{endTime-beginTime}")
        # nb_plot(nbCluster)
        merge_label = np.zeros(len(data), dtype=int)
        k = 0
        for i in nbCluster:
            for j in nbCluster[i].dataIndex:
                merge_label[j] = k
            k += 1

        true_label = df.values[:, 0]
        true_label = true_label.astype(int)
        true_label = np.array(true_label).flatten()

        print("ACC", acc(true_label, merge_label))
        print("NMI", metrics.normalized_mutual_info_score(true_label, merge_label))