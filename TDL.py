# lib of temporal difference learning for othello
import ModifiedBench.Function as Function
import ModifiedBench.Comparator as Comparator
import ModifiedBench.pso as pso
import math
import random
import numpy as np
import copy
import Problem
import matplotlib.pyplot as plt
import Othello

npop = 10
ps = 100
pn = "othello"


def cprint(cond,text):
    if cond:
        print(text)

def tdl_othello():
    max_games = 1e6
    games = 0
    # strategy representation-WPC
    othello = Problem.Problem()
    othello.instantiate(pn)
    player = Othello.Player()
    player.initialize_weight(othello.xlb,othello.xub)  # strategy weight matrix
    print(player.w)
    # while games < max_games:
    pass



def test_selfplay(weight):
    # self-play othello
    othello = Problem.Problem()
    othello.instantiate(pn)
    state = Othello.State()
    state.is_end()
    player = Othello.Player()
    player.w = copy.deepcopy(weight)  # strategy weight matrix
    while not state.end:
        # self-play begins
        print("round",state.round,"turn", state.turn)
        skip, state = player.act(state, update_strategy=False)  # black player moves, symbol = 1
        state.turn = - state.turn  # switch to white player strategy
        player.symbol = state.turn
        print("round", state.round, "turn", state.turn)
        skip, state = player.act(state, update_strategy=False)  # white player moves, symbol = -1
        state.turn = - state.turn  # switch to black player strategy
        player.symbol = state.turn
        state.round += 1
        if state.end:
            if state.winner == 1:
                print("Black wins.")
            elif state.winner == -1:
                print("White wins.")
            else:
                print("End with a draw")


def test_tdl(verbose=0):
    max_games = 3e6
    games = 0
    best_players = []
    # strategy representation-WPC
    othello = Problem.Problem()
    othello.instantiate(pn)
    player = Othello.Player()
    player.initialize_weight(othello.xlb, othello.xub)  # strategy weight matrix
    # training process
    while games < max_games:
        if games % (max_games / 20) == 0:
            best_players.append(copy.deepcopy(player))
        print("game",games)
        state = Othello.State()
        state.is_end()
        while not state.end:
            # self-play begins
            cprint(verbose == 1, "round "+str(state.round)+" turn "+str(state.turn))
            skip, state = player.act(state, display=False)  # black player moves, symbol = 1
            state.turn = - state.turn  # switch to white player strategy
            player.symbol = state.turn
            cprint(verbose == 1, "round "+str(state.round)+" turn "+str(state.turn))
            skip, state = player.act(state, display=False)  # white player moves, symbol = -1
            state.turn = - state.turn  # switch to black player strategy
            player.symbol = state.turn
            state.round += 1
        # print(np.round(player.w,2))
        if state.end and verbose == 1:
            if state.winner == 1:
                print("Black wins.")
            elif state.winner == -1:
                print("White wins.")
            elif state.winner == 0:
                print("End with a draw")
            else:
                print("Invalid winner!")
        games += 1
    print("Training is over.")
    print("Saving the strategy.")
    np.save("./Data/weight_mat/tdl_weight.npy", player.w)


if __name__ == '__main__':
    # test_tdl()
    data = np.load("./Data/opt_process/s0.npy")
    player = Othello.Player()
    player.load_weight(data)
    print(np.round(player.w, 2))
    # test_vs_random_opponents([player],oppo_size=1)
    Othello.test_vs_fixed_opponent(player, Othello.hp)
