import matplotlib.pyplot as plt
import numpy as np

def plot_curve(data):
    x = np.linspace(0,1e6,len(data))
    plt.plot(x,data)

if __name__ == '__main__':
    # retrieve data
    ALG = "TDL_vs_SELF"
    PATH = "./Measure/"
    NRUN = 20
    fig = plt.figure()
    for r in range(NRUN):
        FILENAME = ALG + "-wr-r" + str(r) + ".npy"
        data = np.load(PATH + FILENAME)
        plot_curve(data)
    plt.ylim(0, 1)
    plt.title(ALG + " winning rate versus sample player")
    plt.ylabel("winning rate")
    plt.xlabel("game count")
    plt.show()
