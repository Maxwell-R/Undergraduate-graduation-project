import math
import sys
import numpy as np
import matplotlib.pyplot as plt
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
        b = b + ComR * math.sqrt(K**2 + 1)
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
    for i in range(Blist.__len__() - 1):
        if i % 2 == 0:
            plt.plot([EndX[i], EndX[i+1]],[K * EndX[i] + Blist[i], K * EndX[i + 1] + Blist[i + 1]], c='#4682B4')
            Length += math.sqrt((EndX[i] - EndX[i + 1]) ** 2 + (K * EndX[i] + Blist[i] - K * EndX[i + 1] - Blist[i + 1]) ** 2)
        else:
            plt.plot([BeginX[i], BeginX[i + 1]], [K * BeginX[i] + Blist[i], K * BeginX[i + 1] + Blist[i + 1]], c='#4682B4')
            Length += math.sqrt((BeginX[i] - BeginX[i + 1]) ** 2 + (K * BeginX[i] + Blist[i] - K * BeginX[i + 1] - Blist[i + 1]) ** 2)
    return [BeginX[0], BeginX[0] * K + Blist[0],
            EndX[Blist.__len__() - 1], EndX[Blist.__len__() - 1] * K + Blist[Blist.__len__() - 1]], Length

clusterCen=np.zeros((150,2),dtype=np.float)
for i in range(150):
    #首先得到簇的中心位置
    tempLocCen=np.random.randint(int(100/2),200-int(100/2),(1,2))
    clusterCen[i]=tempLocCen
for i in range(150):
    plt.scatter(clusterCen[i][0],clusterCen[i][1],s=4,c='#4682B4')
Node, Leng = GetLine(150, clusterCen, 1, 10)
print(Leng)
print(Node)
plt.show()