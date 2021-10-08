# adaptive fitness shaping coevolutionary learning
# classical coevolutionary learning algorithm
import multiprocessing as mul
import os
import random
import numpy as np
import copy
import Problem
import matplotlib.pyplot as plt
import PageRank as PR
import Othello
import EvolutionStrategy as ES
import Measure
import EDA

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


def selection(mode,pop,param):
    # (mu, lambda) - evolution strategy wih mu=lambda
    ps = len(pop)
    if mode == "round-robin":
        pass
    elif mode == "k-random-opponent":
        sample_size = param["sample_size"]
        sample_lb = param["sample_lb"]
        sample_ub = param["sample_ub"]
        game_count = 2 * sample_size * ps
        sample = []
        for i in range(sample_size):
            player = Othello.Player()
            player.load_weight(np.random.uniform(sample_lb, sample_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
            sample.append(copy.deepcopy(player))
        wmat, opwmat = interaction(pop, sample, "k-random-opponent") #  generational winning count, opponents' generational winning count
        # fitness shaping
        fit = np.dot(wmat, np.ones(sample_size) * 1.0)
        index = np.argsort(fit)
        next_pop = []
        print("k-rand fit",fit)
        for i in range(int(len(index) / 2), len(index)):
            next_pop.append(copy.deepcopy(pop[index[i]]))
        return next_pop, index[int(len(index) / 2):], game_count
    elif mode == "k-random-opponent-v2":
        sample_size = param["sample_size"]
        sample_lb = param["sample_lb"]
        sample_ub = param["sample_ub"]
        game_count = 2 * sample_size * ps
        sample = []
        for i in range(sample_size):
            player = Othello.Player()
            player.load_weight(np.random.uniform(sample_lb, sample_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
            sample.append(copy.deepcopy(player))
        wmat, opwmat = interaction(pop, sample, "k-random-opponent") #  generational winning count, opponents' generational winning count
        opfit = np.dot(opwmat, np.ones(ps) * 1.0) / (2 * ps)
        print("opfit", opfit)
        # fitness shaping
        fit = np.dot(wmat, opfit)
        index = np.argsort(fit)
        next_pop = []
        print("k-rand fit",fit)
        for i in range(int(len(index) / 2), len(index)):
            next_pop.append(copy.deepcopy(pop[index[i]]))
        return next_pop, index[int(len(index) / 2):], game_count
    elif mode == "pos":
        dop = dict()
        dop["lb"] = param["sample_lb"]
        dop["ub"] = param["sample_ub"]
        next_pop, index, game_count = proc_oriented_selection(pop, dop)
        return next_pop, index, game_count
    else:
        print("Wrong mode type.")
        return

def proc_oriented_selection(pop, dop):
    # distribution of opponent population
    max_opsize = 50
    game_count = 0
    gap = 3
    tpop = copy.deepcopy(pop)
    ps = len(tpop)
    num_winner = ps / 2  # number of winner
    oppo_size = 0
    pi = np.arange(0, ps)
    rank = np.zeros(ps)
    rank_count = 0
    winpoint = np.zeros(ps)
    while True:
        ps = len(tpop)
        if ps == num_winner:
            sorted_index = np.argsort(winpoint)
            rpop = []
            for i in range(len(sorted_index)):
                rpop.append(copy.deepcopy(tpop[sorted_index[i]]))
            pi = copy.deepcopy(pi[sorted_index])
            print("opponent size",oppo_size)
            return rpop, pi, game_count
        # randomly generate a test player according to the distribution
        op = Othello.Player()
        op.load_weight(np.random.uniform(dop["lb"], dop["ub"], (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        oppo_size += 1
        reward = np.zeros(ps)
        opwincount = np.zeros(ps)
        for i in range(ps):
            game_count += 2
            game_res = Othello.play_game(tpop[i], op)
            if game_res == 1:
                reward[i] = 1
            elif game_res == -1:
                opwincount[i] += 1
            game_res = Othello.play_game(op, tpop[i])
            if game_res == -1:
                reward[i] = 1
            elif game_res == 1:
                opwincount[i] += 1
        # reward = reward * np.sum(opwincount) / (2 * ps)
        winpoint = winpoint + reward
        # print(wincount)
        index = np.argsort(winpoint)
        # print("index",index)
        if ps > num_winner and winpoint[index[1]] - winpoint[index[0]] >= gap:
            rank_count += 1
            rank[pi[index[0]]] = rank_count
            del tpop[index[0]]
            winpoint = np.delete(winpoint,index[0])
            pi = np.delete(pi,index[0])
        if oppo_size == max_opsize and ps > num_winner:
            for i in range(int(ps - num_winner)):
                del tpop[index[i]]
                winpoint = np.delete(winpoint, index[i])
                pi = np.delete(pi, index[i])
                index[index > index[i]] = index[index > index[i]] - 1
        print("winpoint",winpoint)
        print("rank",index)


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
    mode = "winrate"
    # Test’s fitness is the weighted number of points it receives for making distinctions. A test makes a distinction
    # for a given pair of candidate solutions if the games it played against them gave different outcomes.
    # Each point awarded for a distinction is weighted by the inverse of the number of tests that made that distinction.
    if mode == "distinction":
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
    elif mode == "winrate":
        opfit = np.zeros(opop_size) * 1.0
        for i in range(opop_size):
            opfit[i] = np.sum(opmat[i,:])
        return opfit
    else:
        print("Wrong mode!")
        return

def AFS_CEL(sample_size=1000, pop_size=20, RESAMPLE=True):
    # adaptive fitness shaping coevolutionary learning
    VERSION = "v2"
    ALG = "AFS-CEL"
    EA = "EDA"
    PATH = "./Data/" + ALG + "/" + VERSION + "/"
    K = 0.1  # scaling constant
    gen = 0
    pop = []
    archive = []  # store elitists
    opop = []
    tpop = []
    opop_size = 50
    tpop_size = 10 * opop_size
    opop_wincount = np.zeros(opop_size) * 1.0
    fsw = np.ones(opop_size) * 1.0  # fitness shaping weight
    init_ub = 0.2
    init_lb = -0.2
    weight_ub = 10.0
    weight_lb = -10.0
    sample_ub = 10.0
    sample_lb = -10.0
    sample = []
    wrg = []
    wrs = []
    MAX_GAMES = 3e6
    true_best_fit = 0  # winning rate
    best_index = -1
    best_fit = -1e25
    best_player = Othello.Player()
    # algorithm parameter settting ======================================
    param = dict()
    param["weight_lb"] = weight_lb
    param["weight_ub"] = weight_ub
    param["mu"] = np.zeros((Othello.BOARD_ROWS, Othello.BOARD_COLS))
    param["sigma"] = np.zeros((Othello.BOARD_ROWS, Othello.BOARD_COLS))
    param["topK"] = 5
    param["alpha"] = 0.1
    # ===================================================================
    # initialization stage
    for i in range(int(pop_size)):
        player = Othello.Player()
        player.load_weight(np.random.uniform(init_lb, init_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        param["mu"] += player.w
        pop.append(copy.deepcopy(player))
    for i in range(opop_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(sample_lb, sample_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        opop.append(copy.deepcopy(player))
    param["mu"] /= pop_size
    for i in range(int(pop_size)):
        param["sigma"] += (pop[i].w - param["mu"]) ** 2
    param["sigma"] = np.sqrt(param["sigma"] / pop_size)
    for i in range(tpop_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(sample_lb, sample_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        tpop.append(copy.deepcopy(player))
    for i in range(sample_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(sample_lb, sample_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        sample.append(copy.deepcopy(player))
    # begin iteration
    while Othello.GAME_COUNT < MAX_GAMES:
        if RESAMPLE or best_fit >= 2 * opop_size * 0.95:
            oppo_incr = 20
            for i in range(oppo_incr):
                player = Othello.Player()
                player.load_weight(np.random.uniform(sample_lb, sample_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
                opop.append(copy.deepcopy(player))
            opop_size += oppo_incr
            best_fit = 0
            RESAMPLE = False
            # opop.clear()
        print("gen", gen, "game count", Othello.GAME_COUNT)
        print("bestfit",best_fit)
        if gen >= 200:
            break
        gen += 1
        # offspring generation
        if EA == "ES":
            for i in range(int(pop_size / 2), pop_size):
                pop.append(copy.deepcopy(pop[int(i - (pop_size / 2))]))
                pop[i].w = np.clip(pop[i].w + K * np.random.uniform(-1.0, 1.0, (Othello.BOARD_COLS, Othello.BOARD_ROWS)),
                                   weight_lb, weight_ub)
        elif EA == "EDA":
            pop = EDA.EDA_iterate(pop, param)
        else:
            print("Wrong EA type.")
            return
        # k-opponent interaction
        wmat, opwmat = interaction(pop, opop, "k-random-opponent")  #  generational winning count, opponents' generational winning count
        # fitness shaping
        fit = np.dot(wmat, np.ones(opop_size) * 1.0)
        opfit = np.dot(opwmat, np.ones(pop_size) * 1.0)
        # elitist preservation strategy
        elit_prev = True
        gen_best_fit = np.max(fit)
        if gen_best_fit > best_fit:
            best_wr = Measure.estimated_generalized_performance(best_player, tpop, "generalization")
            if best_wr > true_best_fit:
                true_best_fit = best_wr
            print("best individual improved.")
            best_fit = np.max(fit)
            best_index = np.argmax(fit)
            best_player = copy.deepcopy(pop[int(best_index)])
            # validation process
            elit_prev = False
        index = None
        if EA == "ES":
            # selection
            index = np.argsort(fit)
            print("fit")
            print(fit[index])
            next_pop = []
            for i in range(int(len(index) / 2), len(index)):
                next_pop.append(copy.deepcopy(pop[index[i]]))
            pop = copy.deepcopy(next_pop)
        elif EA == "EDA":
            param["mu"], param["sigma"], best_index = EDA.EDA_param_update(pop,fit,param)
            print("mu",np.round(param["mu"],2))
            print("sigma",np.round(param["sigma"],2))
        else:
            print("Wrong EA type.")
            return
        # elitism preserving strategy
        if elit_prev:
            randi = np.random.randint(0, len(pop))
            pop[randi] = copy.deepcopy(best_player)
        # best_player = copy.deepcopy(pop[-1])
        # measure performance and save intermediate result
        wrs_gen = []
        wrg_gen = []
        for i in range(1):
            wrs_gen.append(Measure.estimated_generalized_performance(best_player, sample, "specialization"))
            wrg_gen.append(Measure.estimated_generalized_performance(best_player, sample, "generalization"))
        # print("survival individual fitness:", fit[index[int(len(index) / 2):]])
        print("bestp",np.round(best_player.w,2))
        print("best player winning rate versus heuristic player:", wrs_gen)
        print("best player winning rate versus sample player:", wrg_gen)
        # save intermediate result
    print("Evolutionary process is over.")
    print("Saving the strategy.")

def process_AFS(VER,RUNID,param):
    PATH = "/public/home/wushenghao/Project/VFOP/Data/AFS-CEL"
    PATHVER = PATH + "/v" + str(VER)
    PATHRUN = PATHVER + "/" + str(RUNID)
    PATHAS = PATHRUN + "/allstrat"
    PATHGB = PATHRUN + "/gbest"
    ps = param["ps"]
    gen_player = []
    for i in range(1,1000):
        pop = []
        for j in range(int(ps/2)):
            FILENAME = "s" + str(j) + "-g" + str(i)
            if not os.path.exists(PATHAS + "/" + FILENAME):
                break
            else:
                player = Othello.Player()
                player.load_weight(PATHAS + "/" + FILENAME)
                pop.append(copy.deepcopy(player))
        gen_player.append(copy.deepcopy(selection("pos",pop,param)))
    for i in range(len(gen_player)):
        np.save(PATHGB + "/gbs" + str(i) + ".npy", gen_player[i].w)

def test_pos():
    sample_size = 500
    sample = []
    init_ub = 0.2
    init_lb = -0.2
    weight_ub = 10.0
    weight_lb = -10.0
    pop = []
    wrg = []
    dop = dict()
    dop["lb"] = weight_lb
    dop["ub"] = weight_ub
    for i in range(sample_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(weight_lb, weight_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        sample.append(copy.deepcopy(player))
    for i in range(10):
        player = Othello.Player()
        player.load_weight(np.random.uniform(init_lb, init_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        pop.append(copy.deepcopy(player))
    bestp, rank, gc = proc_oriented_selection(pop, dop)
    print("game count", gc)
    print("rank", rank)
    for i in range(len(pop)):
        wrg.append(Measure.estimated_generalized_performance(pop[i], sample, "generalization"))
    print(wrg)

def get_precision(label,true_label):
    cr = 0.0
    for i in range(len(label)):
        if label[i] in true_label:
            cr += 1
    cr /= len(true_label)
    return cr

def get_topN_precision(label,true_label):
    topN = len(true_label)
    cr = []
    for i in range(topN):
        cr.append(get_precision(label[i:],true_label[i:]))
    return cr

def compare_selection():
    init_ub = 10.0
    init_lb = -10.0
    weight_ub = 10.0
    weight_lb = -10.0
    param = dict()
    param["sample_size"] = 50
    param["sample_lb"] = weight_lb
    param["sample_ub"] = weight_ub
    pop = []
    sel_size = 30
    for i in range(sel_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(init_lb, init_ub, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        pop.append(copy.deepcopy(player))
    bestp1, index1, gc1 = selection("k-random-opponent-v2", pop, param)
    # bestp2, index2, gc2 = selection("pos", pop, param)
    param["sample_size"] = 500
    true_bestp, true_index, gc3 = selection("k-random-opponent", pop, param)
    print("k-rand========================")
    print("index", index1)
    print("game count", gc1)
    print("precision", get_topN_precision(index1, true_index))
    print("==============================")
    # print("pos===========================")
    # print("index", index2)
    # print("game count", gc2)
    # print("precision", get_topN_precision(index2, true_index))
    # print("==============================")
    print("true index", true_index)


if __name__ == '__main__':
    # test_tdl()
    # compare_selection()
    novelEA(RESAMPLE=False)
    # compare_selection()
    # data = np.load("./Data/weight_mat/cel_weight.npy")
    # player = Othello.Player()
    # player.load_weight(data)
    # print(np.round(player.w, 2))
    # # test_vs_random_opponents([player],oppo_size=1)
    # Othello.test_vs_fixed_opponent(player,Othello.hp)
