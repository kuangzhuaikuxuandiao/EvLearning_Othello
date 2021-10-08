# adaptive fitness shaping coevolutionary learning
# classical coevolutionary learning algorithm
import multiprocessing as mul
import os
import random
import numpy as np
import copy
import Problem
import Cluster
import PageRank as PR
import Othello
import EvolutionStrategy as ES
import Measure

class Population:
    # problem-specific population class definition
    def __init__(self, weight_lb, weight_ub):
        self.pop = []
        self.ps = 0
        self.wlb = weight_lb
        self.wub = weight_ub

    def initialize(self,init_lb,init_ub):
        for i in range(self.ps):
            player = Othello.Player()
            player.load_weight(np.random.uniform(init_lb, init_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
            self.pop.append(copy.deepcopy(player))

    def add_individual(self,player):
        self.pop.append(copy.deepcopy(player))
        self.ps += 1

    def copy_population(self,cpop):
        self.ps = len(cpop)
        self.pop = []
        for i in range(self.ps):
            self.pop.append(copy.deepcopy(cpop[i]))

    def update(self):
        mode = "ES"
        if mode == "ES":
            K = 0.1
            # offspring generation
            for i in range(self.ps):
                self.pop.append(copy.deepcopy(self.pop[i]))
                self.pop[int(i + self.ps)].w = np.clip(
                    self.pop[int(i + self.ps)].w + K * np.random.uniform(-1.0, 1.0,
                                                                         (Othello.BOARD_COLS, Othello.BOARD_ROWS)),
                    self.wlb, self.wub)
            self.ps = len(self.pop)

    def offspring_production(self):
        # offspring generation
        mode = "ES"
        if mode == "ES":
            K = 0.1
            offspring_pop = []
            for i in range(self.ps):
                offspring_pop.append(copy.deepcopy(self.pop[int(i)]))
                offspring_pop[i].w = np.clip(
                    self.pop[i].w + K * np.random.uniform(-1.0, 1.0, (Othello.BOARD_COLS, Othello.BOARD_ROWS)),
                    self.wlb, self.wub)
            return offspring_pop

def ploy_fit(data):
    # 1-d data polynomial fitting
    deg = 5
    datalen = len(data)
    x = np.arange(0, datalen)
    y = copy.deepcopy(data)
    f = np.polyfit(x, y, deg)
    return f


def ploy_pred(factor, data):
    return np.polyval(factor, data)


def symmetry_index(n, wmat):
    tc = 0.0
    sc = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            tc += 1
            if wmat[i][j] == wmat[j][i]:
                sc += 1
    return sc / tc


def transitivity_index(n, wcmat):
    wmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
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
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                trans = True
                if wmat[i][j] == 1 and wmat[j][k] == 1 and wmat[i][k] == 0:
                    trans = False
                if wmat[i][j] == -1 and wmat[j][k] == -1 and wmat[i][k] == 1:
                    trans = False
                if trans:
                    tran_count += 1
                total_count += 1
    return tran_count / total_count


def estimated_generalized_performance(p, ts=None):
    sample_size = len(ts)
    if isinstance(p, list):
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
    elif isinstance(p, Othello.Player):
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

def interaction(pop, ts, mode):
    if mode == "k-random-opponent":
        K = len(ts)
        ps = len(pop)
        wm = np.zeros((ps,K)) * 1.0
        opwm = np.zeros((K,ps)) * 1.0
        for i in range(ps):
            for j in range(K):
                game_res = Othello.play_game(pop[i], ts[j])
                if game_res == 1:
                    wm[i][j] += 1
                elif game_res == -1:
                    opwm[j][i] += 1
                game_res = Othello.play_game(ts[j], pop[i])
                if game_res == -1:
                    wm[i][j] += 1
                elif game_res == 1:
                    opwm[j][i] += 1
        return wm, opwm
    elif mode == "round-robin":
        ps = len(pop)
        wm = np.zeros((ps,ps)) * 1.0
        for i in range(ps):
            for j in range(i+1,ps):
                game_res = Othello.play_game(pop[i], pop[j])
                if game_res == 1:
                    wm[i][j] += 1
                elif game_res == -1:
                    wm[j][i] += 1
                game_res = Othello.play_game(pop[j], pop[i])
                if game_res == -1:
                    wm[i][j] += 1
                elif game_res == 1:
                    wm[j][i] += 1
        return wm
    else:
        return

def opponent_fitness_assign(opop_size,pop_size,wmat,opmat,inmat):
    # Test’s fitness is the weighted number of points it receives for making distinctions. A test makes a distinction
    # for a given pair of candidate solutions if the games it played against them gave different outcomes.
    # Each point awarded for a distinction is weighted by the inverse of the number of tests that made that distinction.
    opfit = np.zeros(opop_size) * 1.0
    for i in range(pop_size):
        for j in range(i+1, pop_size):
            dist_index = []
            for k in range(opop_size):
                if (wmat[i][k] * opmat[k][j] > 0 and inmat[i][j] >= inmat[j][i]) or (
                        wmat[j][k] * opmat[k][i] > 0 and inmat[j][i] >= inmat[i][j]):
                    dist_index.append(k)
            if len(dist_index) == 0:
                continue
            for index in dist_index:
                opfit[index] += 1.0
    return opfit

def PBIL(pop,fit_assign,prob_type="maximization",**args):
    mu = args["mu"]
    delta = args["delta"]

def multipop_fitness_assign(cep, wmat, pop_index):
    # cep: co-evolutionary population
    # pi: individual's population index
    # wmat: winnning matrix
    yita = 0.1
    eval_size = len(cep)
    fit = np.zeros(eval_size) * 0.0
    for i in range(eval_size):
        pi = pop_index[i]
        # fit[i] = (1 - yita) * np.sum(wmat[i, pop_index != pi]) + yita * np.sum(wmat[i, pop_index == pi])
        fit[i] = np.sum(wmat[i, pop_index != pi])
    return fit

def pop_selection(pop, fit):
    index = np.argsort(fit)
    next_pop = []
    for i in range(int(len(index) / 2), len(index)):
        next_pop.append(copy.deepcopy(pop[index[i]]))
    return next_pop

def MPCEL(sample_size=200, pop_size=50):
    # classical coevolutionary learning
    VERSION = "v0"
    ALG = "MPCEL"
    PATH = "./Data/" + ALG + "/" + VERSION + "/"
    K = 0.1  # scaling constant
    gen = 0
    subpop = []
    init_weight = []
    num_spop = 3  # sub-population
    init_spop_size = int(pop_size / num_spop)
    init_ub = 0.2
    init_lb = -0.2
    weight_ub = 10.0
    weight_lb = -10.0
    sample_ub = 10.0
    sample_lb = -10.0
    search_range = weight_ub - weight_lb
    sample = []
    wrg = []
    wrs = []
    MAX_GAMES = 3e6
    stratn = 0  # strategy number
    best_player = None
    # initialization stage
    for i in range(num_spop):
        subpop.append(Population(weight_lb,weight_ub))
    for i in range(pop_size):
        init_weight.append(np.random.uniform(init_lb, init_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
    # k-means population partition
    kmCluster = Cluster.Cluster("kmeans")
    kmCluster.set_kmeans_param(n=pop_size, k=num_spop, dim=(Othello.BOARD_ROWS, Othello.BOARD_COLS), lb=init_lb,
                               ub=init_ub)
    c_index = kmCluster.partition(init_weight)[1]
    for i in range(pop_size):
        player = Othello.Player()
        player.load_weight(init_weight[i])
        subpop[int(c_index[i])].add_individual(player)
    ps = []
    for i in range(num_spop):
        ps.append(subpop[i].ps)
    print("initial population size",ps)
    # begin iteration
    while Othello.GAME_COUNT < MAX_GAMES:
        sample = []
        for i in range(sample_size):
            player = Othello.Player()
            player.load_weight(np.random.uniform(sample_lb, sample_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
            sample.append(copy.deepcopy(player))
        print("gen", gen, "game count", Othello.GAME_COUNT)
        if gen >= 100:
            break
        gen += 1
        # offspring generation
        for i in range(num_spop):
            subpop[i].update()
        ce_pop = []  # co-evolutionary population including all individual players (parents and offspring) from every population
        pop_index = []  # individual's population index
        for i in range(num_spop):
            for j in range(subpop[i].ps):
                ce_pop.append(copy.deepcopy(subpop[i].pop[j]))
                pop_index.append(i)
        pop_index = np.array(pop_index)
        # round-robin interaction
        wmat = interaction(ce_pop, [], "round-robin")  # winning matrix
        # adaptive fitness shaping
        fit = multipop_fitness_assign(ce_pop, wmat, pop_index)
        # selection
        next_subpop = []
        for i in range(num_spop):
            next_subpop.append(Population(weight_lb,weight_ub))
        for i in range(len(ce_pop)):
            next_subpop[int(pop_index[i])].add_individual(ce_pop[i])
        for i in range(num_spop):
            print(fit[pop_index == i])
            print(next_subpop[i].ps)
            next_subpop[i].copy_population(pop_selection(next_subpop[i].pop, fit[pop_index == i]))
            subpop[i] = copy.deepcopy(next_subpop[i])
        # best player selection
        bp_set = []  # best player set
        for i in range(num_spop):
            bp_set.append(copy.deepcopy(subpop[i].pop[-1]))
        # index = np.argsort(fit)
        # best_player = copy.deepcopy(ce_pop[int(index[-1])])
        wrs_gen = []
        wrg_gen = []
        for i in range(len(bp_set)):
            # measure performance and save intermediate result
            wrs_gen.append(Measure.estimated_generalized_performance(bp_set[i], sample, "specialization"))
            wrg_gen.append(Measure.estimated_generalized_performance(bp_set[i], sample, "generalization"))
        print("best player winning rate versus heuristic player:", wrs_gen)
        print("best player winning rate versus sample player:", wrg_gen)
        # wrs.append(wrg_gen)
        # wrg.append(wrg_gen)
    # np.save(PATH + "wrs.npy",wrs)
    # np.save(PATH + "wrg.npy",wrg)
        # save intermediate result
    print("Evolutionary process is over.")
    print("Saving the strategy.")


if __name__ == '__main__':
    # test_tdl()
    MPCEL()
    # data = np.load("./Data/weight_mat/cel_weight.npy")
    # player = Othello.Player()
    # player.load_weight(data)
    # print(np.round(player.w, 2))
    # # test_vs_random_opponents([player],oppo_size=1)
    # Othello.test_vs_fixed_opponent(player,Othello.hp)
