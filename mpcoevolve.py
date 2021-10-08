import ModifiedBench.Function as Function
import ModifiedBench.Comparator as Comparator
import ModifiedBench.pso as pso
import math
import random
import numpy as np
import copy
import Problem
import matplotlib.pyplot as plt

npop = 10
ps = 100
pn = "othello"

def mpcc():
    # problem definition
    # random generation
    prob = Problem.Problem()
    prob.instantiate(pn)
    mpop = []
    for i in range(ps):
        mpop.append(copy.deepcopy(np.random.uniform()))
    pass