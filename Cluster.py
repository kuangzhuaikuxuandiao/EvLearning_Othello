import math
import random
import numpy as np
import Problem
import copy
import matplotlib.pyplot as plt

def distance(x,y):
    dif = x-y
    return np.sqrt(np.sum(dif*dif))


class Cluster:
    def __init__(self, method):
        self.method = method
        self.param = {}

    def set_kmeans_param(self, n, k, dim, lb, ub):
        if not isinstance(dim,tuple):
            print("Cluster parameter setting is unsuccessful. Dimension should be tuple.")
            return
        self.param["n"] = n
        self.param["k"] = k
        self.param["dim"] = dim
        self.param["lb"] = lb
        self.param["ub"] = ub

    def partition(self, data, verbose=0):
        iter = 0
        if self.method == "kmeans":
            nd = self.param["n"]  # number of data
            nc = self.param["k"]  # number of clusters
            dim = self.param["dim"]  # dimensionality
            ub = self.param["ub"]  # vector, upper bound for the initial points
            lb = self.param["lb"]  # vector, lower bound for the initial points
            # initial cluster center generation
            ctr = []
            clusters = []
            cindex = -1 * np.ones(nd)  # cluster index
            for i in range(nc):
                ctr.append(copy.deepcopy(np.random.uniform(lb,ub,dim)))
                clusters.append([])
            flag = True
            while flag:
                if verbose == 1:
                    print(iter)
                iter += 1
                flag = False
                # clear set
                for i in range(nc):
                    clusters[i] = []
                # cluster assign
                for i in range(nd):
                    ci = -1
                    mdist = 1e25
                    for j in range(nc):
                        tdist = distance(data[i], ctr[j])
                        if tdist < mdist:
                            mdist = tdist
                            ci = j
                    if ci == -1:
                        print("Error occurs.")
                        return
                    if ci != cindex[i]:
                        flag = True
                    cindex[i] = ci
                    clusters[ci].append(copy.deepcopy(data[i]))
                # cluster update
                for i in range(nc):
                    if len(clusters[i]) == 0:
                        continue
                    ctr[i] = np.zeros(dim)
                    for j in range(len(clusters[i])):
                        ctr[i] = ctr[i] + clusters[i][j]
                    ctr[i] = ctr[i] / len(clusters[i])
            return ctr, cindex

def test_kmeans():
    colors = 'bgrcmykbrcmykbgrcmykbgrcmyk'
    kmCluster = Cluster("kmeans")
    N = 50
    K = 5
    DIM = (8,8)
    LB = -0.2
    UB = 0.2
    kmCluster.set_kmeans_param(n=N,k=K,dim=DIM,lb=LB,ub=UB)
    data = []
    # data generation
    for i in range(N):
        data.append(np.random.uniform(LB,UB,DIM))
    ctr,c_index = kmCluster.partition(data)
    print(ctr,c_index)
    # visualization
    # fig = plt.figure()
    # for i in range(N):
    #     plt.scatter(data[i][0],data[i][1],c=colors[int(c_index[i])])
    # for i in range(K):
    #     plt.scatter(ctr[i][0],ctr[i][1],c='k')
    # plt.show()

if __name__ == '__main__':
    test_kmeans()