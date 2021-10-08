# implementation of evolution strategy algorithm for othello
import ModifiedBench.Function as Function
import ModifiedBench.Comparator as Comparator
import ModifiedBench.pso as pso
import random
import numpy as np
import copy
import Problem
import matplotlib.pyplot as plt
import Othello

_mu = 1
_lambda = 30

def EvolStrat(mu=1,lamb=30):
    MAX_GAMES = 3e6
    beta = 0.05
    reward = dict()
    reward["win"] = 1
    reward["tie"] = 0
    reward["lose"] = -2
    pplayer = Othello.Player()  # parent players
    pw = np.zeros((Othello.BOARD_ROWS, Othello.BOARD_COLS))
    pplayer.load_weight(pw)
    stratn = 0  # strategy number
    while True:
        print("game",Othello.GAME_COUNT)
        if Othello.GAME_COUNT >= MAX_GAMES:
            break
        # save temporary data
        if Othello.GAME_COUNT % 60000 < 30 * 29:
            print("save strategy "+str(stratn))
            np.save("./Data/opt_process/s"+str(stratn)+".npy",pplayer.w)
            stratn += 1
        # update child players' weight
        cplayers = []  # set of child players
        for i in range(lamb):
            cw = pplayer.w + np.random.normal(0,1.0/(Othello.BOARD_ROWS * Othello.BOARD_COLS),(Othello.BOARD_ROWS, Othello.BOARD_COLS))
            cplayers.append(Othello.Player())
            cplayers[i].load_weight(weight=cw)
        # round-robin evaluation
        scores = np.zeros(lamb)
        for i in range(lamb):
            for j in range(i+1,lamb):
                game_res = Othello.play_game(cplayers[i], cplayers[j])
                if game_res == 1:
                    scores[i] += reward["win"]
                    scores[j] += reward["lose"]
                elif game_res == 0:
                    scores[i] += reward["tie"]
                    scores[j] += reward["tie"]
                elif game_res == -1:
                    scores[i] += reward["lose"]
                    scores[j] += reward["win"]
                else:
                    print("Invalid game result!")
                    return
                game_res = Othello.play_game(cplayers[j], cplayers[i])
                if game_res == 1:
                    scores[j] += reward["win"]
                    scores[i] += reward["lose"]
                elif game_res == 0:
                    scores[j] += reward["tie"]
                    scores[i] += reward["tie"]
                elif game_res == -1:
                    scores[j] += reward["lose"]
                    scores[i] += reward["win"]
                else:
                    print("Invalid game result!")
                    return
        # find the player with highest score
        bi = -1
        bscore = -1e10
        for i in range(lamb):
            if scores[i] > bscore:
                bi = i
        if bi == -1:
            print("Error occurs")
            return
        # parent-child averaging
        pplayer.w = pplayer.w + beta * (cplayers[bi].w - pplayer.w)
    print("Evolutionary process is over.")
    print("Saving the strategy.")
    np.save("./Data/weight_mat/cel_weight.npy", pplayer.w)
