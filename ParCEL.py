# coevolutionary learning algorithm
import ModifiedBench.Function as Function
import ModifiedBench.Comparator as Comparator
import ModifiedBench.pso as pso
import os
import random
import numpy as np
import copy
import Problem
import matplotlib.pyplot as plt
import PageRank as PR
import Othello
import EvolutionStrategy as ES
import multiprocessing as mul

def func(x,q):
    q.put((x,x+1))

def ploy_fit(data):
    # 1-d data polynomial fitting
    deg = 5
    datalen = len(data)
    x = np.arange(0,datalen)
    y = copy.deepcopy(data)
    f = np.polyfit(x,y,deg)
    return f

def ploy_pred(factor,data):
    return np.polyval(factor,data)

def symmetry_index(n,wmat):
    tc = 0.0
    sc = 0.0
    for i in range(n):
        for j in range(i+1,n):
            tc += 1
            if wmat[i][j] == wmat[j][i]:
                sc += 1
    return sc/tc

def transitivity_index(n,wcmat):
    wmat = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            if wcmat[i][j] > wcmat[j][i]:
                wmat[i][j] = 1
                wmat[j][i] = 0
            elif wcmat[i][j] < wcmat[j][i]:
                wmat[j][i] = 1
                wmat[i][j] = 0
            else:
                wmat[i][j] = 0.5
                wmat[j][i] = 0.5
    total_count = 0.0
    tran_count = 0.0
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                trans = True
                if wmat[i][j] == 1 and wmat[j][k] == 1 and wmat[i][k] == 0:
                    trans = False
                if wmat[i][j] == -1 and wmat[j][k] == -1 and wmat[i][k] == 1:
                    trans = False
                if trans:
                    tran_count += 1
                total_count += 1
    return tran_count / total_count

def estimated_generalized_performance(p,queue,pindex,ts):
    # individual-level measure
    # game_size: specify how many games should play between tested player and heuristic player,
    #            (play black for game_size games and play white for game_size games)
    # p: tested player
    # pindex: player index
    game_size = 100
    sample_size = len(ts)
    if isinstance(p,Othello.Player):
        # generalization performance
        wcg = 0.0
        for j in range(sample_size):
            game_res = Othello.play_game(p, ts[j])
            if game_res == 1:
                wcg += 1
            game_res = Othello.play_game(ts[j], p)
            if game_res == -1:
                wcg += 1
        wrg = wcg * 0.5 / sample_size
        # specialization performance
        wcs = 0.0
        for j in range(game_size):
            game_res = Othello.play_game(p, Othello.hp)
            if game_res == 1:
                wcs += 1
            game_res = Othello.play_game(Othello.hp, p)
            if game_res == -1:
                wcs += 1
        wrs = wcs * 0.5 / game_size
        queue.put([pindex,wrg,wrs])
    else:
        print("Wrong input type!")
        return

def competitive_fitness(mode, pop, eval_scheme="winrate", ts=None):
    # mode: interaction mode (round-robin, k-random-opponent, single elimination..)
    # eval_scheme: evaluation metric (winning rate, pagerank..)
    # ts: test set including opponents
    # wmat: winning matrix
    # first hand\ second hand   1  2  3
    #                        ----------
    #                       1| -1  1  3
    #                       2|  0 -1  2
    #                       3|  3  2 -1
    # -1:self-play, 0:draw, others: winner's index
    # wcmat: win count matrix
    # first hand\ second hand   1  2  3
    #                        ----------
    #                       1|  0  1  3
    #                       2|  9  0  2
    #                       3|  7  8  0
    # winning count versus its opponent
    if mode == "round-robin":
        ps = len(pop)
        wincount = np.zeros(ps)
        wmat = -1.0 * np.ones((ps,ps))  # winner matrix
        wcmat = np.zeros((ps,ps))
        game_graph = PR.Graph(ps)
        # fill the graph with competition results
        for i in range(ps):
            for j in range(i+1, ps):
                game_res = Othello.play_game(pop[i], pop[j])
                if game_res == 1:
                    wmat[i][j] = i
                    wincount[i] += 1
                    game_graph.update_adj_mat(i, j, 1)
                    wcmat[i][j] += 1
                elif game_res == -1:
                    wmat[i][j] = j
                    wincount[j] += 1
                    game_graph.update_adj_mat(j, i, 1)
                    wcmat[j][i] += 1
                else:
                    wmat[i][j] = 0
                    wcmat[i][j] += 0.5
                    wcmat[j][i] += 0.5
                # reverse side
                game_res = Othello.play_game(pop[j], pop[i])
                if game_res == 1:
                    wmat[j][i] = j
                    wincount[j] += 1
                    game_graph.update_adj_mat(j, i, 1)
                    wcmat[j][i] += 1
                elif game_res == -1:
                    wmat[j][i] = i
                    wincount[i] += 1
                    game_graph.update_adj_mat(i, j, 1)
                    wcmat[i][j] += 1
                else:
                    wmat[j][i] = 0
                    wcmat[i][j] += 0.5
                    wcmat[j][i] += 0.5
        if eval_scheme == "pagerank":
            # end point handling
            for i in range(game_graph.num_node):
                if 0.5 and 1 not in game_graph.adj_mat[i]:
                    game_graph.adj_mat[i] = 1.0 * np.ones(game_graph.num_node) / game_graph.num_node
            fit = 1.0 / game_graph.power_method()
        elif eval_scheme == "winrate":
            fit = 0.5 * wincount / (ps - 1)
        else:
            print("Wrong evaluation scheme!")
            return
        # si = symmetry_index(ps,wmat)
        # ti = transitivity_index(ps,wcmat)
    elif mode == "k-random-opponent":
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
        winrate = wincount / K
        fit = copy.deepcopy(winrate)
        return fit
    else:
        print("Invalid interaction mode.")
        return

def novelEA(sample_size=500,pop_size=20):
    # classical coevolutionary learning
    PATH = "/public/home/wushenghao/Project/VFOP/Data/test/v0"
    K = 0.1  # scaling constant
    gen = 0
    pop = []
    sample = []
    MAX_GAMES = 3e6
    best_player = None
    # initialization stage
    for i in range(int(pop_size/2)):
        player = Othello.Player()
        player.load_weight(np.random.uniform(-0.2, 0.2, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        pop.append(copy.deepcopy(player))
    for i in range(sample_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(-0.2, 0.2, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        sample.append(copy.deepcopy(player))
    q = mul.Queue()
    # begin iteration
    while Othello.GAME_COUNT < MAX_GAMES:
        if gen >= 100:
            break
        gen += 1
        # offspring generation
        for i in range(int(pop_size/2),pop_size):
            pop.append(copy.deepcopy(pop[int(i-(pop_size/2))]))
            pop[i].w = np.clip(pop[i].w + K * np.random.uniform(-1.0, 1.0 , (Othello.BOARD_COLS, Othello.BOARD_ROWS)), -10.0, 10.0)
        # round-robin interaction and fitness assignment
        fit = competitive_fitness("round-robin",pop)
        # individual player performance estimation in a distributed computing scheme
        Proc = [mul.Process(target=estimated_generalized_performance, args=(pop[i], q, i, sample)) for i in
                range(pop_size)]
        for proc in Proc:
            proc.start()
        for proc in Proc:
            proc.join()
        # read data from the sub-process by Queue
        wrg = np.zeros(pop_size)
        wrs = np.zeros(pop_size)
        data_est = [q.get() for proc in Proc]
        for i in range(len(data_est)):
            pindex = data_est[i][0]
            wrg[pindex] = data_est[i][1]
            wrs[pindex] = data_est[i][2]
        # save fitness, generalization winning rate, specialization winning rate for each generation
        np.save(PATH + "/fit-g" + str(gen) + ".npy", fit)
        np.save(PATH + "/wrg-g" + str(gen) + ".npy", wrg)
        np.save(PATH + "/wrs-g" + str(gen) + ".npy", wrs)
        # selection
        index = np.argsort(fit)
        next_pop = []
        for i in range(int(len(index) / 2), len(index)):
            next_pop.append(copy.deepcopy(pop[index[i]]))
        pop = copy.deepcopy(next_pop)
        best_player = copy.deepcopy(pop[-1])
    print("Evolutionary process is over.")
    print("Saving the strategy.")

if __name__ == '__main__':
    novelEA()