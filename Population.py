import math
import random
import numpy as np
import copy

class Population:
    def __init__(self, ptype):
        self.ptype = ptype
        self.ps = 0
        self.pop = []
        self.pb = []
        self.gb = None

    def copy_pop(self, cp):
        if self.ptype == "swarm":
            self.pop = copy.deepcopy(cp)
            self.ps = len(cp)
            self.pb = copy.deepcopy(cp)
        else:
            self.pop = copy.deepcopy(cp)
            self.ps = len(cp)

    def add_ind(self, ind, pbi=None):
        if self.ptype == "swarm":
            self.pop.append(copy.deepcopy(ind))
            self.ps = len(self.pop)
            self.pb.append(copy.deepcopy(pbi))
        else:
            self.pop.append(copy.deepcopy(ind))
            self.ps = len(self.pop)

