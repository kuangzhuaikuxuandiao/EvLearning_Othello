# classical coevolutionary learning algorithm
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

def estimated_generalized_performance(p,ts=None):
    sample_size = len(ts)
    if isinstance(p,list):
        psize = len(p)
        wincount = np.zeros(psize) * 1.0
        for i in range(psize):
            for j in range(sample_size):
                game_res = Othello.play_game(p[i], ts[j])
                if game_res == 1:
                    wincount[i] += 1
                game_res = Othello.play_game(ts[j], p[i])
                if game_res == -1:
                    wincount[i] += 1
        winrate = wincount * 0.5 / sample_size
    elif isinstance(p,Othello.Player):
        wincount = 0
        for i in range(sample_size):
            game_res = Othello.play_game(p, ts[i])
            if game_res == 1:
                wincount += 1
            game_res = Othello.play_game(ts[i], p)
            if game_res == -1:
                wincount += 1
        winrate = wincount * 0.5 / sample_size
    else:
        print("Wrong input type!")
        return
    return winrate

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
        return fit
    elif mode == "k-random-opponent":
        K = len(ts)
        opwincount = np.zeros(K)
        ps = len(pop)
        wincount = np.zeros(ps)
        for i in range(ps):
            for j in range(K):
                game_res = Othello.play_game(pop[i], ts[j])
                if game_res == 1:
                    wincount[i] += 1
                elif game_res == -1:
                    opwincount[j] += 1
                game_res = Othello.play_game(ts[j], pop[i])
                if game_res == -1:
                    wincount[i] += 1
                elif game_res == 1:
                    opwincount[j] += 1
        # print(wincount)
        winrate = wincount / K
        fit = copy.deepcopy(winrate)
        return fit
    else:
        print("Invalid interaction mode.")
        return

def novelEA(sample_size=500,pop_size=30):
    # classical coevolutionary learning
    filename = "20-06-28"
    K = 0.1  # scaling constant
    gen = 0
    pop = []
    opop = []
    opop_size = 20
    opweight = np.ones(opop_size) * 1.0
    sample = []
    MAX_GAMES = 3e6
    stratn = 0  # strategy number
    best_player = None
    # initialization stage
    for i in range(int(pop_size/2)):
        player = Othello.Player()
        player.load_weight(np.random.uniform(-0.2, 0.2, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        pop.append(copy.deepcopy(player))
    for i in range(int(pop_size/2)):
        player = Othello.Player()
        player.load_weight(np.random.uniform(-0.2, 0.2, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        pop.append(copy.deepcopy(player))
    for i in range(sample_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(-0.2, 0.2, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        sample.append(copy.deepcopy(player))
    # begin iteration
    while Othello.GAME_COUNT < MAX_GAMES:
        gfn = "gen" + str(gen)
        print("gen", gen, "game count", Othello.GAME_COUNT)
        gen += 1
        if not os.path.exists("./Data/" + filename + "/" + gfn):
            os.mkdir("./Data/" + filename + "/" + gfn)
        # offspring generation
        for i in range(int(pop_size/2),pop_size):
            pop.append(copy.deepcopy(pop[int(i-(pop_size/2))]))
            pop[i].w = np.clip(pop[i].w + K * np.random.uniform(-1.0, 1.0 , (Othello.BOARD_COLS, Othello.BOARD_ROWS)), -10.0, 10.0)
        # round-robin interaction and fitness assignment
        fit = competitive_fitness("round-robin",pop)
        # selection
        index = np.argsort(fit)
        next_pop = []
        for i in range(int(len(index) / 2), len(index)):
            next_pop.append(copy.deepcopy(pop[index[i]]))
        pop = copy.deepcopy(next_pop)
        best_player = copy.deepcopy(pop[-1])
        # save intermediate result
        for i in range(len(pop)):
            stratname = "s"+str(i)+".npy"
            np.save("./Data/" + filename + "/" + gfn + "/" + stratname, pop[i].w)
    print("Evolutionary process is over.")
    print("Saving the strategy.")
    np.save("./Data/"+filename+"/ccl_weight.npy", best_player.w)


def CCL(pop_size=30):
    # classical coevolutionary learning
    filename = "20-06-23"
    K = 0.1  # scaling constant
    gen = 0
    pop = []
    MAX_GAMES = 3e6
    stratn = 0  # strategy number
    best_player = None
    # initialization stage
    for i in range(int(pop_size/2)):
        player = Othello.Player()
        player.load_weight(np.random.uniform(-0.2, 0.2, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        pop.append(copy.deepcopy(player))
    # begin iteration
    while Othello.GAME_COUNT < MAX_GAMES:
        gfn = "gen" + str(gen)
        print("gen", gen, "game count", Othello.GAME_COUNT)
        gen += 1
        if not os.path.exists("./Data/" + filename + "/" + gfn):
            os.mkdir("./Data/" + filename + "/" + gfn)
        # offspring generation
        for i in range(int(pop_size/2),pop_size):
            pop.append(copy.deepcopy(pop[int(i-(pop_size/2))]))
            pop[i].w = np.clip(pop[i].w + K * np.random.uniform(-1.0, 1.0 , (Othello.BOARD_COLS, Othello.BOARD_ROWS)), -10.0, 10.0)
        # round-robin interaction and fitness assignment
        fit = competitive_fitness("round-robin",pop)
        # selection
        index = np.argsort(fit)
        next_pop = []
        for i in range(int(len(index) / 2), len(index)):
            next_pop.append(copy.deepcopy(pop[index[i]]))
        pop = copy.deepcopy(next_pop)
        best_player = copy.deepcopy(pop[-1])
        # save intermediate result
        for i in range(len(pop)):
            stratname = "s"+str(i)+".npy"
            np.save("./Data/" + filename + "/" + gfn + "/" + stratname, pop[i].w)
    print("Evolutionary process is over.")
    print("Saving the strategy.")
    np.save("./Data/"+filename+"/ccl_weight.npy", best_player.w)

if __name__ == '__main__':
    # test_tdl()
    novelEA()
    # data = np.load("./Data/weight_mat/cel_weight.npy")
    # player = Othello.Player()
    # player.load_weight(data)
    # print(np.round(player.w, 2))
    # # test_vs_random_opponents([player],oppo_size=1)
    # Othello.test_vs_fixed_opponent(player,Othello.hp)
