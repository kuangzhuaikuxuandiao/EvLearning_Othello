import numpy as np

# problem scale, users should define fitness assignment function in the main file.
pn = ["othello"]

class Problem:
    def __init__(self):
        self.name = None
        self.encoding_scheme = "realn"
        self.dim = 1
        self.xub = np.zeros(self.dim)
        self.xlb = np.ones(self.dim) * 100.0
        self.param = dict()

    def instantiate(self, name):
        self.name = name
        if self.name == "othello":
            es = "WPC"  # weighted piece counter
            NROW = 8
            NCOL = 8
            self.param["NROW"] = NROW
            self.param["NCOL"] = NCOL
            self.param["ES"] = "WPC"
            if es == "WPC":
                self.dim = NROW * NCOL
                self.xub = np.ones(self.dim).reshape((NROW,NCOL)) * 1.0
                self.xlb = np.ones(self.dim).reshape((NROW,NCOL)) * -1.0
        pass

    def objective_fit(self,ind):
        # absolute fitness
        pass

    def subjective_fit(self,ind):
        # relatvie fitness by individual interaction
        pass

    def performance_metric(self):
        pass