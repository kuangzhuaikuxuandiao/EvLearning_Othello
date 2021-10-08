import math
import random
import numpy as np
import Problem
import copy

class Individual:
    def __init__(self,prob):
        self.prob = prob
        self.x = np.random.uniform(prob.lb,prob.ub,prob.dim)
        self.exfit = None  # external fit
        self.hidfit = None  # hidden fit, inaccessible
        self.infit = 0.0  # internal fitness