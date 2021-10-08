import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import Othello

# 用于存储图
class Graph:
    def __init__(self, N):
        self.num_node = N
        self.linked_node_map = {}  # 邻接表，
        self.PR_map = {}  # 存储每个节点的入度
        self.adj_mat = np.zeros((self.num_node,self.num_node)) * 1.0  # 邻接矩阵

    def import_adj_mat(self, mat):
        self.adj_mat = copy.deepcopy(mat)

    # 更新邻接矩阵
    def update_adj_mat(self, i, j, val):
        self.adj_mat[i][j] = val

    # 添加节点
    def add_node(self, node_id):
        if node_id not in self.linked_node_map:
            self.linked_node_map[node_id] = set({})
            self.PR_map[node_id] = 0
        else:
            print("这个节点已经存在")

    # 增加一个从Node1指向node2的边。允许添加新节点
    def add_link(self, node1, node2):
        if node1 not in self.linked_node_map:
            self.add_node(node1)
        if node2 not in self.linked_node_map:
            self.add_node(node2)
        self.linked_node_map[node1].add(node2)  # 为node1添加一个邻接节点，表示ndoe2引用了node1

    # 计算pr
    def get_PR(self, epoch_num=100, d=0.85):  # 配置迭代轮数，以及阻尼系数
        for i in range(epoch_num):
            for node in self.PR_map:  # 遍历每一个节点
                self.PR_map[node] = (1 - d) + d * sum(
                    [self.PR_map[temp_node] for temp_node in self.linked_node_map[node]])  # 原始版公式
            print(self.PR_map)

    # 幂法求pr
    def power_method(self, epsilon=0.001, d=0.85):
        eps = np.ones(self.num_node) * epsilon
        x = np.ones(self.num_node) / self.num_node
        pm = copy.deepcopy(self.adj_mat)  # 概率转移矩阵
        for i in range(self.num_node):
            # 归一化处理
            if np.sum(pm[:,i]) != 0:
                pm[:,i] = copy.deepcopy(1.0 * pm[:,i] / np.sum(pm[:,i]))
        print(pm)
        # print(pm)
        idm = np.ones((self.num_node,self.num_node)) / self.num_node
        am = d * pm + (1 - d) * idm
        r = np.dot(am, x)
        gen = 0
        while True:
            # print(gen)
            gen += 1
            print(np.round(r,2))
            if np.all(np.abs(x - r) < eps):
                return r
            else:
                x = copy.deepcopy(r)
                r = np.dot(am, x)

# edges = [[1, 2], [2, 3], [2, 4], [3, 4], [4, 3]]  # 模拟的一个网页链接网络
if __name__ == '__main__':
    mat = 1.0 * np.array([[0,1,1,1],
                          [1,0,1,0],
                          [0,0,0,0],
                          [0,1,0,0]])
    graph = Graph(4)
    graph.import_adj_mat(mat)
    print(graph.power_method())
    pass