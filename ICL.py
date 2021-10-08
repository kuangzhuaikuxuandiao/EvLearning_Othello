# coevolutionary learning algorithm
import os
import random
import numpy as np
import copy
import Problem
import PageRank as PR
import Othello
import EvolutionStrategy as ES
import multiprocessing as mul

def competitive_fitness(pop, ts=None):
    K = len(ts)
    ps = len(pop)
    wincount = np.zeros(ps)
    for i in range(ps):
        for j in range(K):
            game_res = Othello.play_game(pop[i], ts[j])
            if game_res == 1:
                wincount[i] += 1
            game_res = Othello.play_game(ts[j], pop[i])
            if game_res == -1:
                wincount[i] += 1
    # print(wincount)
    winrate = 0.5 * wincount / K
    fit = copy.deepcopy(winrate)
    return fit

def ICL(runID,rnd_seed,sample_size=50,pop_size=20):
    np.random.seed(rnd_seed)
    # classical coevolutionary learning
    ALG = "ICL" + str(sample_size)
    # PATH = "/public/home/wushenghao/Project/CEL/Data/" + ALG + "/" + str(runID)
    # if not os.path.exists(PATH):
    #     os.mkdir(PATH)
    # PATHAS = PATH + "/allstrat"
    # PATHGB = PATH + "/gbest"
    # if not os.path.exists(PATHAS):
    #     os.mkdir(PATHAS)
    # if not os.path.exists(PATHGB):
    #     os.mkdir(PATHGB)
    init_ub = 1.0
    init_lb = -1.0
    weight_ub = 10.0
    weight_lb = -10.0
    sample_ub = 10.0
    sample_lb = -10.0
    K = 0.1  # scaling constant
    gen = 0
    pop = []
    sample = []
    MAX_GAMES = 3e6
    best_player = None
    # initialization stage
    for i in range(int(pop_size/2)):
        player = Othello.Player()
        player.load_weight(np.random.uniform(init_lb, init_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        pop.append(copy.deepcopy(player))
    for i in range(sample_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(sample_lb, sample_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        sample.append(copy.deepcopy(player))
    # begin iteration
    while Othello.GAME_COUNT < MAX_GAMES:
        gen += 1
        # offspring generation
        for i in range(int(pop_size/2),pop_size):
            pop.append(copy.deepcopy(pop[int(i-(pop_size/2))]))
            pop[i].w = np.clip(pop[i].w + K * np.random.uniform(-1.0, 1.0, (Othello.BOARD_COLS, Othello.BOARD_ROWS)),
                               weight_lb, weight_ub)
        # save intermediate result
        # for i in range(len(pop)):
        #     stratname = "s" + str(i) + "-g" + str(gen) + ".npy"
        #     np.save(PATHAS + "/" + stratname, pop[i].w)
        # round-robin interaction and fitness assignment
        fit = competitive_fitness(pop,ts=sample)
        # selection
        index = np.argsort(fit)
        next_pop = []
        for i in range(int(len(index) / 2), len(index)):
            next_pop.append(copy.deepcopy(pop[index[i]]))
        pop = copy.deepcopy(next_pop)
        best_player = copy.deepcopy(pop[-1])
        print(fit)
    # np.save(PATH + "/" + ALG + "_weight.npy", best_player.w)

if __name__ == '__main__':
    # test_tdl()
    # seed = np.random.uniform(0,20000,1000)
    # NRUN = 20
    # Proc = [mul.Process(target=ICL,args=(i,seed[i],)) for i in range(NRUN)]
    # for p in Proc:
    #     p.start()
    # for p in Proc:
    #     p.join()
    ICL(0,0)
    # data = np.load("./Data/weight_mat/cel_weight.npy")
    # player = Othello.Player()
    # player.load_weight(data)
    # print(np.round(player.w, 2))
    # # test_vs_random_opponents([player],oppo_size=1)
    # Othello.test_vs_fixed_opponent(player,Othello.hp)
