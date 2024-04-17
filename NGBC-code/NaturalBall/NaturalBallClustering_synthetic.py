import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class NaturalBall:
    def __init__(self, data, label, dataIndex):
        self.data = data
        self.dataIndex = dataIndex
        self.center = self.data.mean(0)
        self.radius = self.get_radius()
        self.label = label
        self.num = len(data)
        self.out = 0
        self.size = 1

    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)


def spilt_ball(data, index, dense, sparse):
    ball1 = []
    ball2 = []
    A = pdist(data)
    D = squareform(A)
    # r, c = np.where(D == np.max(D))  # where返回满足条件的点的坐标，是一个有两行的矩阵，第一行是横坐标，第二行是纵坐标
    r1 = index.index(dense)  # 取出当前数据集中距离最远的两个点在数据集中的下标，点 1
    c1 = index.index(sparse)  # 取出当前数据集中距离最远的两个点在数据集中的下标，点 2
    # 以上是算数据集的距离矩阵，涉及线性知识点
    for j in range(0, len(data)):  # 遍历整个距离矩阵D，如果某个数据距离点1近就归为点1的球中，若距离点2近，就归为点2的球中
        if D[j, r1] < D[j, c1]:
            ball1.append(index[j])
        else:
            ball2.append(index[j])
    # 返回数据下标
    if len(ball1) > len(ball2):
        return set(ball1), set(ball2)
    else:
        return set(ball2), set(ball1)


def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diffMat = np.tile(center, (num, 1)) - hb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius


def plot_dot(data):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o')


def nb_plot(gbs):
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        25: '#ff8444',
        26: '#F0F8FF',
        
    }

    plt.figure(figsize=(10, 10))
    label_num = {}
    for i in range(0, len(gbs)):
        label_num.setdefault(gbs[i].label, 0)
        label_num[gbs[i].label] = label_num.get(gbs[i].label) + len(gbs[i].data)

    label = set()
    for key in label_num.keys():
        label.add(key)

    list = []
    for i in range(0, len(label)):
        list.append(label.pop())

    for i in range(0, len(list)):
        if list[i] == -1:
            list.remove(-1)
            break

    for key in gbs.keys():
        for i in range(0, len(list)):
            if gbs[key].label == list[i]:
                if i < 173:
                    plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                                marker='o')
                else:
                    plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c='blue', linewidths=5, alpha=0.9,
                                marker='o')
    plt.show()


def draw_ball(hb_list):
    for data in hb_list:
        if len(data) > 1:
            center = data.mean(0)
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            # 使用参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            # 参数ls是球的线类型，‘-’指实线，lw表示线宽
            plt.plot(x, y, ls="-", color="black", lw=0.7, alpha=0.5)
    plt.show()


def form_nb_cluster_by_neighbor(data, dis_index, nbGroup):

    # nbGroup里面的key是某个球，value是某个球里面的数据点
    connected_nb_cluster = nbGroup

    def merge_by_neighbor(connected_nb_cluster):

        connected_temp_nb_cluster = {}
        iterateList = [False] * len(connected_nb_cluster)
        clusterCount = 0
        for i in range(len(iterateList)):
            if not iterateList[i]:
                connected_temp_nb_cluster.setdefault(clusterCount, connected_nb_cluster[i])
                for j in range(i, len(iterateList)):
                    if (i >= len(data)) != (j >= len(data)):
                        if len(connected_temp_nb_cluster[clusterCount] & connected_nb_cluster[j]) > 1 and not iterateList[j]:
                            iterateList[j] = True
                            connected_temp_nb_cluster[clusterCount] |= connected_nb_cluster[j]
                    elif (i < len(data)) & (j < len(data)):
                        if len(connected_temp_nb_cluster[clusterCount] & connected_nb_cluster[j]) > 0 and not iterateList[j]:
                            iterateList[j] = True
                            connected_temp_nb_cluster[clusterCount] |= connected_nb_cluster[j]
                clusterCount += 1
        return connected_temp_nb_cluster

    potentialNoise = set()

    # 进行公共邻居的合并：只要两个球有公共点就进行合并
    while True:
        connected_temp_nb_cluster = merge_by_neighbor(connected_nb_cluster)

        if len(connected_nb_cluster) == len(connected_temp_nb_cluster):
            for i in range(len(connected_nb_cluster)):
                # 最终形成的某个簇中如果数据点太少，则当做噪声点在后面进行分配
                if len(connected_nb_cluster[i]) < len(data) ** 0.5:
                    potentialNoise |= connected_nb_cluster[i]
                    connected_nb_cluster.pop(i)
            break
        connected_nb_cluster = connected_temp_nb_cluster

    # 分配潜在的噪声点到各个簇
    data_label_list = [-1]*len(data)
    all_cluster_data = set()
    for key, value in connected_nb_cluster.items():
        all_cluster_data |= value
        for v in value:
            data_label_list[v] = key

    all_data_set = set(range(len(data)))
    potentialNoise |= (all_data_set-all_cluster_data)

    # copy一个已分配标签用于对照，避免数据点的分配问题
    data_label_list_for_compare = data_label_list.copy()

    # 对潜在的噪声数据点进行打标签，离哪个已经被打标的数据点最近，就打哪个标签的值
    for noise in potentialNoise:

        # 取出index在距离矩阵中最近的那个点的标签
        for index in dis_index[noise][1]:
            if data_label_list_for_compare[index] != -1:
                data_label_list[noise] = data_label_list_for_compare[index]
                break

    # 把新打标的潜在噪声数据点分配到实际簇中
    for i, label in enumerate(data_label_list):
        connected_nb_cluster[label].add(i)

    # 根据数据的标签分配到最终实际的簇
    nbCluster = {}
    labelCount = 0
    for value in connected_nb_cluster.values():
        nbData = []
        if len(value) > 0:
            for v in value:
                nbData.append(data[v])
            nbData = np.array(nbData)
            nbCluster[labelCount] = NaturalBall(nbData, labelCount, list(value))
            labelCount += 1

    return nbCluster


if __name__ == '__main__':
    print("This is the main of NaturalBallClustering_synthetic.py")