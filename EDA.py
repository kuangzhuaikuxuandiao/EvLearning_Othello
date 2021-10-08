import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import Othello
'''
PBILc for Coevolutionary learning
'''

def EDA_param_update(pop,fit,param):
    mu = param["mu"]
    sigma = param["sigma"]
    topK = param["topK"]   # 0.5 * ps
    alpha = param["alpha"]   # 0.5
    index = np.argsort(-fit)
    mu = (1 - alpha) * mu + alpha * (pop[index[0]].w + pop[index[1]].w - pop[index[-1]].w)
    sig_elitist = np.zeros((Othello.BOARD_ROWS, Othello.BOARD_COLS))
    weight_avg = np.zeros((Othello.BOARD_ROWS, Othello.BOARD_COLS))
    best_index = index[0]
    for i in range(topK):
        weight_avg = weight_avg + pop[i].w
    weight_avg = weight_avg / topK
    for i in range(topK):
        sig_elitist += (pop[i].w - weight_avg) ** 2
    sig_elitist = np.sqrt(sig_elitist / topK)
    sigma = (1-alpha) * sigma + alpha * sig_elitist
    return mu,sigma,best_index


def EDA_iterate(pop,param):
    mu = param["mu"]
    sigma = param["sigma"]
    weight_lb = param["weight_lb"]
    weight_ub = param["weight_ub"]
    for i in range(len(pop)):
        pop[i].w = np.clip(np.random.normal(loc=mu,scale=sigma),weight_lb,weight_ub)
    return pop