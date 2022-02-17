"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
from gym.envs.classic_control import rendering
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
# 簇头节点聚类路由协议
# def ClusterProtocol(协议类型,簇序号,最大仿真时长,每个簇的节点个数)
def ClusterProtocol(type, clusterInd, maxTimeSlot, eachClusterDeviceNum,seed):
    np.random.seed(seed)
    allTimeClusterHeadInd = np.zeros((int(maxTimeSlot / 100 - 1), 2), dtype=np.int)
    if type == 'Leach':
        Tn = np.zeros((eachClusterDeviceNum,), dtype=np.float)
        for timeSlot in range(100, maxTimeSlot, 100):
            if (sum(Tn) == -eachClusterDeviceNum):
                # 当只剩下一个节点未当选时，这个节点一定当选，并更新Tn。
                Tn = np.zeros((eachClusterDeviceNum,), dtype=np.float)
            Round = int(timeSlot / 100)  # LEACH定义了“轮”(round)的概念,一轮由初始化和稳定工作两个阶段组成。
            P = 1 / eachClusterDeviceNum  # 为节点中成为聚头的百分数
            T = P / (1 - P * (Round % (1 / P)))  # 阈值
            for i in range(eachClusterDeviceNum):
                if Tn[i] != -1:
                    Tn[i] = np.random.rand()
            maxInd = np.where(Tn == max(Tn))
            allTimeClusterHeadInd[int(timeSlot / 100) - 1][0] = timeSlot
            allTimeClusterHeadInd[int(timeSlot / 100) - 1][1] = maxInd[0][0]
            Tn[maxInd[0][0]] = -1
        return allTimeClusterHeadInd
    if type == 'Teen':
        return allTimeClusterHeadInd
#tsp问题精确求解器
class TSPSolution:
    def __init__(self,X,start_node):
        self.X = X #距离矩阵
        self.start_node = start_node #开始的节点
        self.array = [[0]*(2**len(self.X)) for i in range(len(self.X))] #记录处于x节点，未经历M个节点时，矩阵储存x的下一步是M中哪个节点
    def transfer(self,sets):
        su = 0
        for s in sets:
            su = su + 2**s # 二进制转换
        return su
    # tsp总接口
    def tsp(self):
        s = self.start_node
        num = len(self.X)
        cities = list(range(num)) #形成节点的集合
        past_sets = [s] #已遍历节点集合
        cities.pop(cities.index(s)) #构建未经历节点的集合
        node = s #初始节点
        return self.solve(node,cities) #求解函数
    def solve(self,node,future_sets):
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
            copy.pop(i) # 删除第i个节点，认为已经完成对其的访问
            distance.append(self.X[node][s_i] + self.solve(s_i,copy))
        # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 未遍历节点集合
        c = self.transfer(future_sets)
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.array[node][c] = next_one
        return d
class CartPoleEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.times=0
        self.seedNum = 120  # 随机种子数 120
        np.random.seed(self.seedNum)  # 初始化随机数生成器
        self.areaLength = 1000  # 仿真空间边长
        self.clusterLength = 100  # 簇的边长
        self.eachClusterDeviceNum = 25  # 每一个簇的设备有25个
        self.clusterNum = 10  # 一共有10个簇
        self.maxTimeSlot = 30000  # 模拟进行的时间单位长度
        self.UAV_v=10 #UAV飞行速度10米每秒【参考文献？】
        self.energy=2 #耗能系数【参考文献？】
        self.UAVCommunRange=20 #无人机通信半径20米【参考文献？】
        self.clusterCen = np.zeros((self.clusterNum, 2), dtype=np.float) #记录簇中心
        self.clusterOne=np.zeros((self.clusterNum, 2), dtype=np.float) #记录序号1
        # 初始化每一个簇的每个设备的位置
        self.clusterLoc = np.zeros((self.clusterNum, self.eachClusterDeviceNum, 2), dtype=np.float)
        # 部署设备分配位置
        for i in range(self.clusterNum):
            # 首先得到簇的中心位置
            tempLocCen = np.random.randint(int(self.clusterLength / 2), self.areaLength - int(self.clusterLength / 2),
                                           (1, 2))
            self.clusterCen[i][0]=tempLocCen[0][0]
            self.clusterCen[i][1]=tempLocCen[0][1]
            self.clusterOne[i][0]=self.clusterLoc[i][1][0]
            self.clusterOne[i][1]=self.clusterLoc[i][1][1]
            # 然后部署簇剩余设备位置
            for k in range(self.eachClusterDeviceNum):
                self.clusterLoc[i][k][0] = np.random.randint(tempLocCen[0][0] - int(self.clusterLength / 2),
                                                             tempLocCen[0][0] + int(self.clusterLength / 2))
                self.clusterLoc[i][k][1] = np.random.randint(tempLocCen[0][1] - int(self.clusterLength / 2),
                                                             tempLocCen[0][1] + int(self.clusterLength / 2))
        # 计算随时间变化后各簇的簇头的序号
        #ans = ClusterProtocol('Leach', 0, self.maxTimeSlot, self.eachClusterDeviceNum)
        #self.clusterHeadInd = np.zeros((self.clusterNum, np.size(ans, 0), 2), dtype=np.int)  # 每个簇的第几个节点是簇头
        #for i in range(clusterNum):
            # 得到不同时间timeslot下的簇头节点的序号
        #    self.clusterHeadInd[i] = ClusterProtocol('Leach', i, self.maxTimeSlot, self.eachClusterDeviceNum)
        # 通过clusterCen得到每个簇之间的距离矩阵DisMat
        DisMat = np.zeros((self.clusterNum, self.clusterNum), dtype=np.float)
        for i in range(self.clusterNum):
            for j in range(self.clusterNum):
                if i == j:
                    DisMat[i][j] = -1
                else:
                    DisMat[i][j] = math.sqrt(
                        (self.clusterCen[i][0] - self.clusterCen[j][0]) ** 2 + (self.clusterCen[i][1] - self.clusterCen[j][1]) ** 2)
        D = DisMat
        S = TSPSolution(D, 0)
        S.tsp()
        # 开始回溯
        M = S.array
        lists = list(range(len(S.X)))
        start = S.start_node
        self.UAVVisitExactOrder = []
        while len(lists) > 0:
            lists.pop(lists.index(start))
            m = S.transfer(lists)
            next_node = S.array[start][m]
            self.UAVVisitExactOrder.append(start)
            start = next_node


        self.action_space = spaces.Box(low=1,high=100,shape=(self.clusterNum*4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0,high=1000,shape=(self.clusterNum,2), dtype=np.float32)
        self.seed()
        self.viewer=rendering.Viewer(1000,1000)
        self.state = [[500,500],[500,500],[500,500],[500,500],[500,500],[500,500],[500,500],[500,500],[500,500],[500,500]]
        self.beginLoc=self.state[self.UAVVisitExactOrder[0]]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        for i in range(self.clusterNum):
            self.state[i]=np.average([[0,1000],[1000,0],[0,0],[1000,1000]],axis=0,weights=action[i*4:i*4+4])
        # print("目前悬停点为",self.state)
        beginLoc=self.state[self.UAVVisitExactOrder[0]]
        # print("出发点为",beginLoc,"是第",self.UAVVisitExactOrder[0],"个簇")
        disTotal = 0
        time=0
        obs=np.zeros((self.clusterNum,2),dtype=np.float)
        sumR=np.zeros((self.clusterNum,),dtype=np.float)
        for i in range(np.size(self.UAVVisitExactOrder,0)):
            disTotal=disTotal+math.sqrt((self.state[self.UAVVisitExactOrder[i]][0]-beginLoc[0])**2+
                                        (self.state[self.UAVVisitExactOrder[i]][1]-beginLoc[1])**2)
            # print("第",self.UAVVisitExactOrder[i],"个簇的悬停点是",self.state[self.UAVVisitExactOrder[i]],"与出发点",beginLoc,"的距离为",math.sqrt((self.state[self.UAVVisitExactOrder[i]][0]-beginLoc[0])**2+
            #                             (self.state[self.UAVVisitExactOrder[i]][1]-beginLoc[1])**2))
            ch=np.random.randint(0,self.eachClusterDeviceNum,1)

            obs[i][0]=self.clusterLoc[self.UAVVisitExactOrder[i]][ch[0]][0]
            obs[i][1]=self.clusterLoc[self.UAVVisitExactOrder[i]][ch[0]][1]
            # print("第",self.UAVVisitExactOrder[i],"个簇的簇中心为",obs[i])
            dis=math.sqrt((self.clusterLoc[self.UAVVisitExactOrder[i]][ch[0]][0]-self.state[self.UAVVisitExactOrder[i]][0])**2+
                          (self.clusterLoc[self.UAVVisitExactOrder[i]][ch[0]][1]-self.state[self.UAVVisitExactOrder[i]][1])**2)
            # print("与悬停点之间的位置为",dis)
            if dis<self.UAVCommunRange:
                beginLoc=self.state[self.UAVVisitExactOrder[i]]
                sumR[self.UAVVisitExactOrder[i]]=10
                # print("保持停点为",beginLoc)

            else:
                beginLoc=self.clusterLoc[self.UAVVisitExactOrder[i]][ch[0]]
                disTotal=disTotal+dis
                sumR[self.UAVVisitExactOrder[i]]=-dis/10
                # print("修改悬停点为" , beginLoc)

        # reward=-disTotal
        reward=sum(sumR)
        # print("当前action的reward为",reward)
        done=False
        self.times=self.times+1

        if self.times>5:
            done=True
            self.times=0
        return np.array(obs, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = [[500,500],[500,500],[500,500],[500,500],[500,500],[500,500],[500,500],[500,500],[500,500],[500,500]]
        return self.state

    def render(self, mode="human"):
        for i in range(10):
            line=rendering.Line((0,i*100),(self.areaLength,i*100))
            self.viewer.add_geom(line)
            line=rendering.Line((i*100,0),(i*100,self.areaLength))
            self.viewer.add_geom(line)
        line1=rendering.Line((0,1000),(1000,1000))
        line2=rendering.Line((1000,0),(1000,1000))
        self.viewer.add_geom(line1)
        self.viewer.add_geom(line2)
        for i in range(self.clusterNum):
            for j in range(self.eachClusterDeviceNum):
                circle=rendering.make_circle(5)
                circle_transform=rendering.Transform(translation=self.clusterLoc[i][j])
                circle.add_attr(circle_transform)
                self.viewer.add_geom(circle)
        for i in range(self.clusterNum):
            self.viewer.draw_circle(8, color=(1, 0, 0)).add_attr(
            rendering.Transform(translation=(self.state[i][0], self.state[i][1])))

        return self.viewer.render(return_rgb_array=mode == 'human')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

