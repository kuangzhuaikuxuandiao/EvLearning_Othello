# peformance measure
import Othello
import numpy as np
import multiprocessing as mul
import copy
import matplotlib.pyplot as plt
import os
import AFS_CEL

def estimated_generalized_performance(p,ts,mode):
    # individual-level measure
    # game_size: specify how many games should play between tested player and heuristic player,
    #            (play black for game_size games and play white for game_size games)
    # p: tested player
    # pindex: player index
    game_size = 100
    sample_size = len(ts)
    if isinstance(p,Othello.Player):
        # generalization performance
        if mode == "generalization":
            wcg = 0.0
            for j in range(sample_size):
                game_res = Othello.play_game(p, ts[j])
                if game_res == 1:
                    wcg += 1
                game_res = Othello.play_game(ts[j], p)
                if game_res == -1:
                    wcg += 1
            wrg = wcg * 0.5 / sample_size
            return wrg
        # specialization performance
        elif mode == "specialization":
            wcs = 0.0
            for j in range(game_size):
                game_res = Othello.play_game(p, Othello.hp)
                if game_res == 1:
                    wcs += 1
                game_res = Othello.play_game(Othello.hp, p)
                if game_res == -1:
                    wcs += 1
            wrs = wcs * 0.5 / game_size
            return wrs
    else:
        print("Wrong input type!")
        return

if __name__ == '__main__':
    # test_tdl()
    sample_size = 100
    sample = []
    for i in range(sample_size):
        player = Othello.Player()
        player.load_weight(np.random.uniform(-10.0, 10.0, (Othello.BOARD_ROWS, Othello.BOARD_COLS)))
        sample.append(copy.deepcopy(player))
    ALG = "AFS-CEL"
    wrg = []
    for gen in range(1,214):
        player = Othello.Player()
        player.load_weight(np.load("./Data/AFS-CEL/v1/gbs-g" + str(gen) + ".npy"))
        wr = estimated_generalized_performance(player,sample,"generalization")
        print("generalization winning rate:",wr)
        wrg.append(wr)
    np.save("./Data/AFS-CEL/v1/wrg-all.npy",wrg)
    plt.figure()
    plt.plot(np.arange(0,len(wrg)),wrg)
    plt.show()
    # begin = 0
    # end = 20
    # proc = [mul.Process(target=process_gbest_data,args=(ALG,"v0",i,sample)) for i in range(begin,end)]
    # for p in proc:
    #     p.start()
    # for p in proc:
    #     p.join()