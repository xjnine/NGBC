import numpy as np
from NaturalBall.NaturalBallClustering_synthetic import get_radius, draw_ball, plot_dot, spilt_ball


class NNSearch:

    def __init__(self, A):
        self.A = A

    # 获得带有索引的排序字典dis_index
    def get_dis_index(self):
        A = self.A
        n = A.shape[0]
        dis_index = {}
        nn = {}
        rnn = {}
        for i in range(0, n):
            dis = np.sort(A[i, :])
            index = np.argsort(A[i, :])
            dis_index[i] = [dis, index]
            nn[i] = []
            rnn[i] = []
        return dis_index, nn, rnn

    # 自动迭代寻找自然邻居，返回迭代次数t,最近邻nn，逆近邻rnn
    def natural_search(self):
        n = self.A.shape[0]
        dis_index, nn, rnn = self.get_dis_index()
        nb = [0] * n
        t = 0
        num_1 = 0
        num_2 = 0
        while t < n:
            for i in range(0, n):
                x = i
                y = dis_index[x][1][t + 1]
                nn[x].append(y)
                rnn[y].append(x)
                nb[y] = nb[y] + 1
            num_1 = nb.count(0)
            if num_1 != num_2:
                num_2 = num_1
            else:
                break
            t = t + 1
        return t, nn, rnn, dis_index


    def get_nb_group(self, nn, rnn, data):

        # 获取邻居和反邻居的交集列表
        intersectionList = []
        for i in range(len(nn)):
            ithSet = set(nn[i]+[i]) & set(rnn[i] + [i])
            intersectionList.append(ithSet)
        # 初始：将每个交集【球】封装到字典,字典里面的value是数据下标
        nbGroup = {}
        ballCount = 0
        for ball in intersectionList:
            nbGroup[ballCount] = ball
            ballCount += 1

        # 将只有数据下标的球转换为实际数据的球
        def fillData(nbGroup):
            initialBall = []
            for key, value in nbGroup.items():
                ball = []
                for v in value:
                    ball.extend([data[v, :]])
                ball = np.array(ball)
                initialBall.extend([ball])
            return initialBall

        def splitNB(ball):
            intersectDict = {}
            for b in ball:
                intersectDict[b] = len(intersectionList[b])
            sorted_intersect_dict = sorted(intersectDict.items(),key=lambda x:x[1])
            ballData = []
            for b in ball:
                ballData.extend([data[b, :]])

            ballData = np.array(ballData)

            originBall, newBall = spilt_ball(ballData, list(ball), sorted_intersect_dict[-1][0],
                                             sorted_intersect_dict[0][0])

            return originBall, newBall

        # 得到初始球，填充数据计算半径
        refinedBall = fillData(nbGroup)
        # 画出初始未分裂的球分布
        plot_dot(data)
        draw_ball(refinedBall)

        radiusList = [0] * len(refinedBall)
        for i, nb in enumerate(refinedBall):
            radiusList[i] = get_radius(nb)
        # 计算平均半径值时忽略半径值为0(单点球)的球，更符合逻辑，且避免死循环

        radiusAray = np.array(radiusList)
        radiusMean = radiusAray.mean()
        radiusStd = np.std(radiusAray)
        radius = radiusMean + radiusStd

        while True:

        #######################################################
            nb_group_len = len(nbGroup)
            for i in range(len(radiusList)):
                if radiusList[i] > radius:
                    # 更新group字典
                    (originBall, newBall) = splitNB(nbGroup[i])
                    nbGroup[i] = originBall
                    if len(newBall) > 0:
                        nbGroup[ballCount] = newBall
                        ballCount += 1

            # 更新数据
            refinedBall = fillData(nbGroup)

            # 更新半径列表
            radiusList = [0] * len(refinedBall)
            for i, nb in enumerate(refinedBall):
                radiusList[i] = get_radius(nb)
            # 判断所有球是否小于半径两倍
            radiusJudge = [i <= radius for i in radiusList]
            nb_group_len_new = len(nbGroup)

            if all(radiusJudge) or nb_group_len == nb_group_len_new:
                break

        finallBall = refinedBall
        # 画出分裂后的球分布
        # plot_dot(data)
        # draw_ball(finallBall)

        return nbGroup


if __name__ == '__main__':
    print('this is a main process')
