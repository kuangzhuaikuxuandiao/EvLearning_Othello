# -*- coding: utf-8 -*-
import random
import numpy as np
import deap.benchmarks.gp
import gp
import os
import math
import TicTacToe.TicTacToe
import Othello
import matplotlib.pyplot as plt
import copy
import Measure

test_board=np.array([[0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,-1,1,0,0,0],
                     [0,0,0,1,-1,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0]])

def func(x):
    return x**2

def func1(target,args):
    return target(args)

def exampletree():
    # if arg[0] > 3:
    #   return arg[1] + 5
    # else:
    #   return arg[1] - 2
    return gp.node(
        gp.ifw, [
            gp.node(gp.gtw, [gp.paramnode(0), gp.constnode(3)]),
            gp.node(gp.addw, [gp.paramnode(1), gp.constnode(5)]),
            gp.node(gp.subw, [gp.paramnode(1), gp.constnode(2)])
        ]
    )


def userinfo(**user):
    print(user)
    for item in user.items():
        print(item)

def cprint(cond,text):
    if cond:
        print(text)

def test_func(a,b,c=1,**kwargs):
    print(a)
    print(b)
    print(c)
    print(kwargs["d"])

def convert_npy2txt():
    size = 109
    mat = np.zeros((20, size))
    alg = "ICL100"
    metric = "vsBENCH"
    for i in range(20):
        data = np.load("./Measure/" + metric + "/" + alg + "-wr-r" + str(i) + ".npy")
        mat[i] = data
    ub = np.zeros(size)
    lb = np.zeros(size)
    ctr = np.zeros(size)
    for i in range(size):
        ctr[i] = np.mean(mat[:, i])
        std = np.std(mat[:, i])
        lb[i] = ctr[i] - std
        ub[i] = ctr[i] + std
    np.savetxt("./Measure/" + metric + "/" + alg + "-lb.txt", lb)
    np.savetxt("./Measure/" + metric + "/" + alg + "-ub.txt", ub)
    np.savetxt("./Measure/" + metric + "/" + alg + "-mean.txt", ctr)

if __name__ == '__main__':
    data = np.load("E:\Wushenghao\Projects\CEL\Measure\\vsSAMPLE\HEUR-wr.npy")
    print(data)
    data = np.load("E:\Wushenghao\Projects\CEL\Measure\\vsSAMPLE\BENCH-wr.npy")
    print(data)
    data = np.load("E:\Wushenghao\Projects\CEL\Measure\\vsSAMPLE\CTDL-wr.npy")
    print(data)
