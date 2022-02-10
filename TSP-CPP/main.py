# TSP-CPP
import math
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize


# 设置实验参数
def fun(x, a, b):
    return a * x + b


def dis(a, b, c, d):
    return math.sqrt((a - c) ** 2 + (b - d) ** 2)


def GetLine(NodeNum, Node, K, ComR):
    Length = 0
    bmin = sys.maxsize
    bmax = -sys.maxsize
    for i in range(NodeNum):
        if Node[i][1] - K * Node[i][0] < bmin:
            bmin = Node[i][1] - K * Node[i][0]
        if Node[i][1] - K * Node[i][0] > bmax:
            bmax = Node[i][1] - K * Node[i][0]
    Blist = []
    b = bmin
    while True:
        if b > bmax:
            break
        Blist.append(b)
        b = b + ComR * math.sqrt(K ** 2 + 1)
    BeginX = []  # 方程i的x开始位置
    EndX = []  # 方程i的x结束位置
    for i in range(Blist.__len__()):
        BeginXTemp = 1000
        EndXTemp = 0
        for n in range(NodeNum):
            if abs(K * Node[n][0] + Blist[i] - Node[n][1]) / math.sqrt(K * K + 1) < ComR:
                b = Node[n][1] + 1 / K * Node[n][0]
                x = (b - Blist[i]) / (K + 1 / K)
                if x < BeginXTemp:
                    BeginXTemp = x
                if x > EndXTemp:
                    EndXTemp = x
        if BeginXTemp == 1000:
            BeginXTemp = Node[1][0]
        if EndXTemp == 0:
            EndXTemp = Node[2][0]
        BeginX.append(BeginXTemp)
        EndX.append(EndXTemp)

        x1 = np.arange(BeginX[i], EndX[i], 0.01)
        plt.plot(x1, K * x1 + Blist[i], c='#4682B4')
        Length += math.sqrt((BeginX[i] - EndX[i]) ** 2 + (K * BeginX[i] + Blist[i] - K * EndX[i] - Blist[i]) ** 2)
        flag = 0
    for i in range(Blist.__len__() - 1):
        if i % 2 == 0:
            plt.plot([EndX[i], EndX[i + 1]], [K * EndX[i] + Blist[i], K * EndX[i + 1] + Blist[i + 1]], c='#4682B4')
            Length += math.sqrt(
                (EndX[i] - EndX[i + 1]) ** 2 + (K * EndX[i] + Blist[i] - K * EndX[i + 1] - Blist[i + 1]) ** 2)
            flag = 1
        else:
            plt.plot([BeginX[i], BeginX[i + 1]], [K * BeginX[i] + Blist[i], K * BeginX[i + 1] + Blist[i + 1]],
                     c='#4682B4')
            Length += math.sqrt(
                (BeginX[i] - BeginX[i + 1]) ** 2 + (K * BeginX[i] + Blist[i] - K * BeginX[i + 1] - Blist[i + 1]) ** 2)
            flag = 0
    if flag == 1:
        return [BeginX[0], BeginX[0] * K + Blist[0],
                BeginX[Blist.__len__() - 1], BeginX[Blist.__len__() - 1] * K + Blist[Blist.__len__() - 1]], Length
    if flag == 0:
        return [BeginX[0], BeginX[0] * K + Blist[0],
                EndX[Blist.__len__() - 1], EndX[Blist.__len__() - 1] * K + Blist[Blist.__len__() - 1]], Length


seedNum = 820  # 随机种子数 120
np.random.seed(seedNum)  # 初始化随机数生成器
NodeLength = 1000  # 边长1000
NodeNum = 50  # 簇节点数量
clusterNum = 8  # 10个簇
CommonR = 20  # 通信半径10米
clusterLength = 100  # 簇的边长
TotalDis = 0
Loc = np.zeros((NodeNum, 2), np.float32)
clusterCen = np.zeros((clusterNum, 2), dtype=np.float)
# 初始化每一个簇的每个设备的位置
clusterLoc = np.zeros((clusterNum, NodeNum, 2), dtype=np.float)
# 部署设备分配位置
for i in range(clusterNum):
    # 首先得到簇的中心位置
    tempLocCen = np.random.randint(int(clusterLength / 2), NodeLength - int(clusterLength / 2), (1, 2))
    clusterCen[i] = tempLocCen
    # 然后部署簇剩余设备位置
    for k in range(NodeNum):
        clusterLoc[i][k][0] = np.random.randint(tempLocCen[0][0] - int(clusterLength / 2),
                                                tempLocCen[0][0] + int(clusterLength / 2))
        clusterLoc[i][k][1] = np.random.randint(tempLocCen[0][1] - int(clusterLength / 2),
                                                tempLocCen[0][1] + int(clusterLength / 2))
        plt.scatter(clusterLoc[i][k][0], clusterLoc[i][k][1], s=4, c='#4682B4')

# 通过clusterCen得到每个簇之间的距离矩阵DisMat
DisMat = np.zeros((clusterNum, clusterNum), dtype=np.float)
for i in range(clusterNum):
    for j in range(clusterNum):
        if i == j:
            DisMat[i][j] = -1
        else:
            DisMat[i][j] = math.sqrt(
                (clusterCen[i][0] - clusterCen[j][0]) ** 2 + (clusterCen[i][1] - clusterCen[j][1]) ** 2)


# tsp问题精确求解器
class TSPSolution:
    def __init__(self, X, start_node):
        self.X = X  # 距离矩阵
        self.start_node = start_node  # 开始的节点
        self.array = [[0] * (2 ** len(self.X)) for i in range(len(self.X))]  # 记录处于x节点，未经历M个节点时，矩阵储存x的下一步是M中哪个节点

    def transfer(self, sets):
        su = 0
        for s in sets:
            su = su + 2 ** s  # 二进制转换
        return su

    # tsp总接口
    def tsp(self):
        s = self.start_node
        num = len(self.X)
        cities = list(range(num))  # 形成节点的集合
        past_sets = [s]  # 已遍历节点集合
        cities.pop(cities.index(s))  # 构建未经历节点的集合
        node = s  # 初始节点
        return self.solve(node, cities)  # 求解函数

    def solve(self, node, future_sets):
        # 迭代终止条件，表示没有了未遍历节点，直接连接当前节点和起点即可
        if len(future_sets) == 0:
            return self.X[node][self.start_node]
        d = 99999
        # node如果经过future_sets中节点，最后回到原点的距离
        distance = []
        # 遍历未经历的节点
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            copy = future_sets[:]
            copy.pop(i)  # 删除第i个节点，认为已经完成对其的访问
            distance.append(self.X[node][s_i] + self.solve(s_i, copy))
        # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 未遍历节点集合
        c = self.transfer(future_sets)
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.array[node][c] = next_one
        return d


D = DisMat

S = TSPSolution(D, 0)
S.tsp()
# 开始回溯
M = S.array
lists = list(range(len(S.X)))
start = S.start_node
UAVVisitExactOrder = []
while len(lists) > 0:
    print(lists.index(start))
    lists.pop(lists.index(start))
    m = S.transfer(lists)
    next_node = S.array[start][m]
    UAVVisitExactOrder.append(start)
    start = next_node
print(UAVVisitExactOrder)

xielv = np.zeros((clusterNum,))
for i in range(clusterNum - 1):
    xielvtemp = (clusterCen[UAVVisitExactOrder[i]][0] - clusterCen[UAVVisitExactOrder[i + 1]][0]) / (
            clusterCen[UAVVisitExactOrder[i]][1] - clusterCen[UAVVisitExactOrder[i + 1]][1])
    xielv[UAVVisitExactOrder[i]] = xielvtemp
xielv[UAVVisitExactOrder[UAVVisitExactOrder.__len__() - 1]] = (clusterCen[UAVVisitExactOrder[
    UAVVisitExactOrder.__len__() - 1]][0] - clusterCen[UAVVisitExactOrder[0]][0]) / (clusterCen[UAVVisitExactOrder[
    UAVVisitExactOrder.__len__() - 1]][1] - clusterCen[UAVVisitExactOrder[0]][1])

BeginAndEnd = []
for i in range(clusterNum):
    BAE, Dis = GetLine(NodeNum, clusterLoc[i], -1 / xielv[i], CommonR)
    TotalDis += Dis
    BeginAndEnd.append(BAE)
for i in range(clusterNum - 1):
    # dis1 = dis(BeginAndEnd[UAVVisitExactOrder[i]][0], BeginAndEnd[UAVVisitExactOrder[i]][1],
    #           BeginAndEnd[UAVVisitExactOrder[i + 1]][0], BeginAndEnd[UAVVisitExactOrder[i + 1]][1])
    # dis2 = dis(BeginAndEnd[UAVVisitExactOrder[i]][0], BeginAndEnd[UAVVisitExactOrder[i]][1],
    #           BeginAndEnd[UAVVisitExactOrder[i + 1]][2], BeginAndEnd[UAVVisitExactOrder[i + 1]][3])
    dis3 = dis(BeginAndEnd[UAVVisitExactOrder[i]][2], BeginAndEnd[UAVVisitExactOrder[i]][3],
               BeginAndEnd[UAVVisitExactOrder[i + 1]][0], BeginAndEnd[UAVVisitExactOrder[i + 1]][1])
    # dis4 = dis(BeginAndEnd[UAVVisitExactOrder[i]][2], BeginAndEnd[UAVVisitExactOrder[i]][3],
    #           BeginAndEnd[UAVVisitExactOrder[i + 1]][2], BeginAndEnd[UAVVisitExactOrder[i + 1]][3])
    # if(min(dis1, dis2, dis3, dis4) == dis1):
    #    plt.plot([BeginAndEnd[UAVVisitExactOrder[i]][0],BeginAndEnd[UAVVisitExactOrder[i+1]][0]],
    #             [BeginAndEnd[UAVVisitExactOrder[i]][1],BeginAndEnd[UAVVisitExactOrder[i+1]][1]],c='#4682B4')
    #    TotalDis += dis1
    # if (min(dis1, dis2, dis3, dis4) == dis2):
    #    plt.plot([BeginAndEnd[UAVVisitExactOrder[i]][0], BeginAndEnd[UAVVisitExactOrder[i + 1]][2]],
    #             [BeginAndEnd[UAVVisitExactOrder[i]][1], BeginAndEnd[UAVVisitExactOrder[i + 1]][3]], c='#4682B4')
    #    TotalDis += dis2
    # if (min(dis1, dis2, dis3, dis4) == dis3):
    plt.plot([BeginAndEnd[UAVVisitExactOrder[i]][2], BeginAndEnd[UAVVisitExactOrder[i + 1]][0]],
             [BeginAndEnd[UAVVisitExactOrder[i]][3], BeginAndEnd[UAVVisitExactOrder[i + 1]][1]], c='#4682B4')
    TotalDis += dis3
    # if (min(dis1, dis2, dis3, dis4) == dis4):
    #    plt.plot([BeginAndEnd[UAVVisitExactOrder[i]][2], BeginAndEnd[UAVVisitExactOrder[i + 1]][2]],
    #             [BeginAndEnd[UAVVisitExactOrder[i]][3], BeginAndEnd[UAVVisitExactOrder[i + 1]][3]], c='#4682B4')
    #    TotalDis += dis4

print(TotalDis)
plt.show()
