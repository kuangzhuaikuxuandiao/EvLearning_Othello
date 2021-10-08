import numpy as np
import copy
import Problem

BOARD_ROWS = 8
BOARD_COLS = 8
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
GAME_COUNT = 0
init_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1, 1, 0, 0, 0],
                       [0, 0, 0, 1, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]])

heuristic_weight = np.array([[1.00, -0.25, 0.10, 0.05, 0.05, 0.10, -0.25, 1.00],
                             [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
                             [0.10, 0.01, 0.05, 0.02, 0.02, 0.05, 0.01, 0.10],
                             [0.05, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.05],
                             [0.05, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.05],
                             [0.10, 0.01, 0.05, 0.02, 0.02, 0.05, 0.01, 0.10],
                             [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
                             [1.00, -0.25, 0.10, 0.05, 0.05, 0.10, -0.25, 1.00]])

benchmark_weight = np.array([[4.622507, -1.477853, 1.409644, -0.066975, -0.305214, 1.633019, -1.050899, 4.365550],
                             [-1.329145, -2.245663, -1.060633, -0.541089, -0.332716, -0.475830, -2.274535, -0.032595],
                             [2.681550, -0.906628, 0.229372, 0.059260, -0.150415, 0.321982, -1.145060, 2.986767],
                             [-0.746066, -0.317389, 0.140040, -0.045266, 0.236595, 0.158543, -0.720833, -0.131124],
                             [-0.305566, -0.328398, 0.073872, -0.131472, -0.172101, 0.016603, -0.511448, -0.264125],
                             [2.777411, -0.769551, 0.676483, 0.282190, 0.007184, 0.269876, -1.408169, 2.396238],
                             [-1.566175, -3.049899, -0.637408, -0.077690, -0.648382, -0.911066, -3.329772, -0.870962],
                             [5.046583, -1.468806, 1.545046, -0.031175, 0.263998, 2.063148, -0.148002, 5.781035]])

ctdl_weight = np.array([[1.02, -0.27, 0.55, -0.10, 0.08, 0.47, -0.38, 1.00],
                        [-0.13, -0.52, -0.18, -0.07, -0.18, -0.29, -0.68, -0.44],
                        [0.55, -0.24, 0.02, -0.01, -0.01, 0.10, -0.13, 0.77],
                        [-0.10, -0.10, 0.01, -0.01, 0.00, -0.01, -0.09, -0.05],
                        [0.05, -0.17, 0.02, -0.04, -0.03, 0.03, -0.09, -0.05],
                        [0.56, -0.25, 0.05, 0.02, -0.02, 0.17, -0.35, 0.42],
                        [-0.25, -0.71, -0.24, -0.23, -0.08, -0.29, -0.63, -0.24],
                        [0.93, -0.44, 0.55, 0.22, -0.15, 0.74, -0.57, 0.97]])

def tanh(x):
    return 2.0 / (1 + np.exp(-2.0 * x)) - 1


def sum_mul(mat1, mat2):
    return np.sum(mat1 * mat2)


class State:
    def __init__(self, input_board=init_board):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        self.round = 0  # the rounds of moves been played
        self.turn = 1  # indicate whose turn to move, symbol = 1, black moves first
        self.data = input_board
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):
                if i == -1:
                    i = 2
                self.hash_val = self.hash_val * 3 + i
        return int(self.hash_val)

    def legal_move(self, xi, xj, symbol):
        if self.data[xi][xj] == 0:
            next_board = copy.deepcopy(self.data)
            flag = False
            # horizontal-east
            ccont = 0  # count the contiguous sequence of opponent's pieces
            for j in range(xj + 1, BOARD_COLS):
                if (self.data[xi][j] == symbol and ccont == 0) or (self.data[xi][j] == 0 and ccont == 0):
                    break
                else:
                    if self.data[xi][j] == -symbol:
                        ccont += 1
                    elif self.data[xi][j] == symbol:
                        next_board[xi][xj] = symbol
                        # turnover the piece
                        for j1 in range(xj + 1, j):
                            next_board[xi][j1] = -next_board[xi][j1]
                        flag = True
                        break
                    else:
                        break
            # horizontal-west
            ccont = 0  # count the contiguous sequence of opponent's pieces
            for j in reversed(range(xj)):
                if (self.data[xi][j] == symbol and ccont == 0) or (self.data[xi][j] == 0 and ccont == 0):
                    break
                else:
                    if self.data[xi][j] == -symbol:
                        ccont += 1
                    elif self.data[xi][j] == symbol:
                        next_board[xi][xj] = symbol
                        # turnover the piece
                        for j1 in range(xj - 1, j, -1):
                            next_board[xi][j1] = -next_board[xi][j1]
                        flag = True
                        break
                    else:
                        break
            # vertical-south
            ccont = 0  # count the contiguous sequence of opponent's pieces
            for i in range(xi + 1, BOARD_ROWS):
                if (self.data[i][xj] == symbol and ccont == 0) or (self.data[i][xj] == 0 and ccont == 0):
                    break
                else:
                    if self.data[i][xj] == -symbol:
                        ccont += 1
                    elif self.data[i][xj] == symbol:
                        next_board[xi][xj] = symbol
                        # turnover the piece
                        for i1 in range(xi + 1, i):
                            next_board[i1][xj] = -next_board[i1][xj]
                        flag = True
                        break
                    else:
                        break
            # vertical-north
            ccont = 0  # count the contiguous sequence of opponent's pieces
            for i in reversed(range(xi)):
                if (self.data[i][xj] == symbol and ccont == 0) or (self.data[i][xj] == 0 and ccont == 0):
                    break
                else:
                    if self.data[i][xj] == -symbol:
                        ccont += 1
                    elif self.data[i][xj] == symbol:
                        next_board[xi][xj] = symbol
                        # turnover the piece
                        for i1 in range(xi - 1, i, -1):
                            next_board[i1][xj] = -next_board[i1][xj]
                        flag = True
                        break
                    else:
                        break
            # diagonal-southeast
            ccont = 0  # count the contiguous sequence of opponent's pieces
            ofs = 1  # offset when indexing board data
            while True:
                # exceeding the matrix boundary
                if (xi + ofs >= BOARD_COLS) or (xj + ofs >= BOARD_ROWS):
                    break
                if (self.data[xi + ofs][xj + ofs] == symbol and ccont == 0) or (
                        self.data[xi + ofs][xj + ofs] == 0 and ccont == 0):
                    break
                else:
                    if self.data[xi + ofs][xj + ofs] == -symbol:
                        ccont += 1
                    elif self.data[xi + ofs][xj + ofs] == symbol:
                        next_board[xi][xj] = symbol
                        # turnover the piece
                        for ofs1 in range(1, ofs):
                            next_board[xi + ofs1][xj + ofs1] = -next_board[xi + ofs1][xj + ofs1]
                        flag = True
                        break
                    else:
                        break
                ofs += 1
            # diagonal-northwest
            ccont = 0  # count the contiguous sequence of opponent's pieces
            ofs = 1  # offset when indexing board data
            while True:
                # exceeding the matrix boundary
                if (xi - ofs < 0) or (xj - ofs < 0):
                    break
                if (self.data[xi - ofs][xj - ofs] == symbol and ccont == 0) or (
                        self.data[xi - ofs][xj - ofs] == 0 and ccont == 0):
                    break
                else:
                    if self.data[xi - ofs][xj - ofs] == -symbol:
                        ccont += 1
                    elif self.data[xi - ofs][xj - ofs] == symbol:
                        next_board[xi][xj] = symbol
                        # turnover the piece
                        for ofs1 in range(1, ofs):
                            next_board[xi - ofs1][xj - ofs1] = -next_board[xi - ofs1][xj - ofs1]
                        flag = True
                        break
                    else:
                        break
                ofs += 1
            # diagonal-southwest
            ccont = 0  # count the contiguous sequence of opponent's pieces
            ofs = 1  # offset when indexing board data
            while True:
                # exceeding the matrix boundary
                if (xi + ofs >= BOARD_ROWS) or (xj - ofs < 0):
                    break
                if (self.data[xi + ofs][xj - ofs] == symbol and ccont == 0) or (
                        self.data[xi + ofs][xj - ofs] == 0 and ccont == 0):
                    break
                else:
                    if self.data[xi + ofs][xj - ofs] == -symbol:
                        ccont += 1
                    elif self.data[xi + ofs][xj - ofs] == symbol:
                        next_board[xi][xj] = symbol
                        # turnover the piece
                        for ofs1 in range(1, ofs):
                            next_board[xi + ofs1][xj - ofs1] = -next_board[xi + ofs1][xj - ofs1]
                        flag = True
                        break
                    else:
                        break
                ofs += 1
            # diagonal-southwest
            ccont = 0  # count the contiguous sequence of opponent's pieces
            ofs = 1  # offset when indexing board data
            while True:
                # exceeding the matrix boundary
                if (xi - ofs < 0) or (xj + ofs >= BOARD_COLS):
                    break
                if (self.data[xi - ofs][xj + ofs] == symbol and ccont == 0) or (
                        self.data[xi - ofs][xj + ofs] == 0 and ccont == 0):
                    break
                else:
                    if self.data[xi - ofs][xj + ofs] == -symbol:
                        ccont += 1
                    elif self.data[xi - ofs][xj + ofs] == symbol:
                        next_board[xi][xj] = symbol
                        # turnover the piece
                        for ofs1 in range(1, ofs):
                            next_board[xi - ofs1][xj + ofs1] = -next_board[xi - ofs1][xj + ofs1]
                        flag = True
                        break
                    else:
                        break
                ofs += 1
            if not flag:
                next_board = None
            return flag, next_board
        else:
            return False, None

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        self.end = True
        cp1 = 0  # count the pieces for the player 1
        cp2 = 0  # count the pieces for the player 1
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.data[i][j] == 0:
                    flag1 = self.legal_move(i, j, 1)[0]
                    flag2 = self.legal_move(i, j, -1)[0]
                    if flag1 or flag2:
                        self.end = False
                        return self.end
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.data[i][j] == 1:
                    cp1 += 1
                elif self.data[i][j] == -1:
                    cp2 += 1
        if self.end:
            if cp1 > cp2:
                self.winner = 1
            elif cp1 < cp2:
                self.winner = -1
            else:
                self.winner = 0
        return self.end

    def display_board(self):
        if self.end:
            winner = "black" if self.winner == 1 else "white"
            print("-------------------")
            for i in range(8):
                row_str = "| "
                for j in range(8):
                    if self.data[i][j] == 0:
                        row_str += "· "
                    elif self.data[i][j] == 1:
                        row_str += "b "
                    elif self.data[i][j] == -1:
                        row_str += "w "
                print(row_str + "|")
            print("-------------------")
            print("game ends:",winner,"wins.")
            print()
        else:
            sturn = "black" if self.turn == 1 else "white"
            print("round:",self.round,"turn:",sturn)
            print("-------------------")
            for i in range(8):
                row_str = "| "
                for j in range(8):
                    if self.data[i][j] == 0:
                        row_str += "· "
                    elif self.data[i][j] == 1:
                        row_str += "b "
                    elif self.data[i][j] == -1:
                        row_str += "w "
                print(row_str + "|")
            print("-------------------")
            print()

class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    # @feedback: if True, both players will receive rewards when game is end
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    # @print: if True, print each board during the game


# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, es="WPC", symbol=1, step_size=0.1, epsilon=0.0, alpha=0.01):
        self.es = es
        self.symbol = symbol
        self.step_size = step_size
        self.epsilon = epsilon
        self.w = None
        self.alpha = alpha
        if self.es == "WPC":
            self.w = np.zeros((BOARD_ROWS, BOARD_COLS))
        else:
            raise Exception("Invalid strategy encoding scheme.")

    def load_weight(self, weight):
        self.w = copy.deepcopy(weight)

    # choose an action based on the state
    def act(self, state, update_strategy=True, display=True):
        skip = True
        val = []
        moves = []
        boards = []
        next_state = copy.deepcopy(state)
        seli = int()
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                is_legal, next_board = state.legal_move(i, j, self.symbol)
                if state.data[i][j] == 0 and is_legal:
                    skip = False
                    val.append(tanh(np.sum(next_board * self.w)))
                    boards.append(next_board)
                    moves.append([i, j, self.symbol])
        # print(moves)
        # print(val)
        if not skip:
            # epsilon greedy
            greed = True
            if np.random.random() > self.epsilon:
                greed = False
                if self.symbol == 1:
                    # select the max action-value, if there are multiple actions, randomly choose one
                    max_val = max(val)
                    indexes = []
                    for k in range(len(val)):
                        if val[k] == max_val:
                            indexes.append(k)
                    ii = np.random.randint(0, len(indexes))
                    seli = indexes[ii]
                elif self.symbol == -1:
                    min_val = min(val)
                    indexes = []
                    for k in range(len(val)):
                        if val[k] == min_val:
                            indexes.append(k)
                    ii = np.random.randint(0, len(indexes))
                    seli = indexes[ii]
            else:
                seli = np.random.randint(0, len(val))
            next_state.data = copy.deepcopy(boards[seli])
            next_state.is_end()
            if update_strategy and not greed:
                self.w = self.update_weight(next_state, state)
        return skip, next_state

    def initialize_weight(self, lb, ub):
        self.w = np.random.uniform(lb, ub, (BOARD_ROWS, BOARD_COLS))

    def update_weight(self, next_state, cur_state):
        if not next_state.end:
            weight = copy.deepcopy(self.w)
            v0 = tanh(sum_mul(cur_state.data, self.w))
            v1 = tanh(sum_mul(next_state.data, self.w))
            weight = weight + self.alpha * (v1 - v0) * (1 - v0 ** 2) * cur_state.data
        else:
            weight = copy.deepcopy(self.w)
            v0 = tanh(sum_mul(cur_state.data, self.w))
            r = 0.0
            if next_state.winner == 1:
                r = 1.0
            elif next_state.winner == -1:
                r = -1.0
            else:
                r = 0.0
            weight = weight + self.alpha * (r - v0) * (1 - v0 ** 2) * cur_state.data
        return weight


hp = Player()  # heuristic player
hp.load_weight(heuristic_weight)

bp = Player()  # benchmark player
bp.load_weight(benchmark_weight)

cp = Player()  # best evolved player by ctdl
cp.load_weight(ctdl_weight)

def train(epochs):
    pass


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play_vs_AI(weight, turn=1):
    state = State()
    state.is_end()
    playerAI = Player()
    playerAI.load_weight(weight)
    playerAI.epsilon = 0.1
    human_turn = turn
    state.display_board()
    if human_turn == 1:
        playerAI.symbol = -1
        move = input("Make your move:")
        if move != "skip":
            move = move.split()
            move = [int(i) for i in move]
            flag, next_board = state.legal_move(move[0], move[1], human_turn)
            while not flag:
                move = input("Illegal move! Try again:")
                move = move.split()
                move = [int(i) for i in move]
                flag, next_board = state.legal_move(move[0], move[1], human_turn)
            state.data = copy.deepcopy(next_board)
            state.turn = - state.turn
            state.is_end()
    else:
        playerAI.symbol = 1
    state.display_board()
    while not state.end:
        # game begins
        skip, state = playerAI.act(state, update_strategy=False)  # black player moves, symbol = 1
        state.turn = -state.turn
        state.is_end()
        state.display_board()
        if state.is_end():
            return
        state.round += 1
        move = input("Make your move:")
        if move != "skip":
            move = move.split()
            move = [int(i) for i in move]
            print(move[0],move[1])
            flag, next_board = state.legal_move(move[0], move[1], human_turn)
            while not flag:
                move = input("Illegal move! Try again:")
                move = move.split()
                move = [int(i) for i in move]
                flag, next_board = state.legal_move(move[0], move[1], human_turn)
            state.data = copy.deepcopy(next_board)
            state.is_end()
        state.turn = - state.turn
        state.display_board()
        if state.is_end():
            return


def play_game(p1, p2, verbose=0, flag1=False, flag2=False):
    global GAME_COUNT
    GAME_COUNT += 1
    state = State()
    state.is_end()
    p1.symbol = 1
    p2.symbol = -1
    while not state.end:
        # self-play begins
        skip, state = p1.act(state, update_strategy=flag1, display=False)  # black player moves, symbol = 1
        state.turn = - state.turn  # switch to white player strategy
        skip, state = p2.act(state, update_strategy=flag2, display=False)  # white player moves, symbol = -1
        state.turn = - state.turn  # switch to black player strategy
        state.round += 1
    return state.winner


def self_play(splayer):
    global GAME_COUNT
    GAME_COUNT += 1
    # self-play othello
    state = State()
    state.is_end()
    print(splayer.w)
    state.display_board()
    while not state.end:
        # self-play begins
        skip, state = splayer.act(state)  # black player moves, symbol = 1
        state.turn = - state.turn  # switch to white player strategy
        splayer.symbol = state.turn
        print(splayer.w)
        state.display_board()
        skip, state = splayer.act(state)  # white player moves, symbol = -1
        state.turn = - state.turn  # switch to black player strategy
        splayer.symbol = state.turn
        print(splayer.w)
        state.display_board()
        state.round += 1


def test_vs_fixed_opponent(tp, fp):
    verbose = 1
    game_size = 500
    winrate = 0.0
    wincount = 0
    tiecount = 0
    losecount = 0
    # player-black
    for j in range(game_size):
        # print(j)
        game_res = play_game(tp, fp)
        if game_res == 1:
            winrate += 1.0
            wincount += 1
        elif game_res == 0:
            tiecount += 1
        else:
            losecount += 1
    # player-white
    for j in range(game_size):
        # print(j)
        game_res = play_game(fp, tp)
        if game_res == -1:
            winrate += 1.0
            wincount += 1
        elif game_res == 0:
            tiecount += 1
        else:
            losecount += 1
    winrate /= 2 * game_size
    return winrate


def test_vs_random_opponents(players, oppo_size=int(1e2), fixed=True):
    data_wr = []
    data_rp = []
    othello = Problem.Problem()
    othello.instantiate("othello")
    weights = []
    Opponents = []
    game_size = 100
    reward = dict()
    reward["win"] = 3
    reward["tie"] = 1
    reward["lose"] = 0
    for i in range(oppo_size):
        weights.append(np.random.uniform(othello.xlb, othello.xub))
        Opponents.append(Player())
        Opponents[i].load_weight(copy.deepcopy(weights[i]))
    pi = 0
    for player in players:
        pi += 1
        rpoint = 0.0  # rating points - performance measure
        winrate = 0.0  # win rate - performance measure
        for i in range(oppo_size):
            wincount = 0
            tiecount = 0
            losecount = 0
            # player-black
            for j in range(game_size):
                game_res = play_game(player, Opponents[i])
                if game_res == 1:
                    winrate += 1.0
                    wincount += 1
                elif game_res == 0:
                    tiecount += 1
                else:
                    losecount += 1
            # player-white
            for j in range(game_size):
                game_res = play_game(Opponents[i], player)
                if game_res == -1:
                    winrate += 1.0
                    wincount += 1
                elif game_res == 0:
                    tiecount += 1
                else:
                    losecount += 1
            # comparison with opponent
            if wincount > losecount:
                rpoint += reward["win"]
            elif wincount == losecount:
                rpoint += reward["tie"]
            else:
                rpoint += reward["lose"]
        winrate /= 2.0 * game_size * oppo_size
        data_wr.append(winrate)
        data_rp.append(rpoint)
    return data_wr, data_rp


if __name__ == '__main__':
    # play_vs_AI(np.zeros((BOARD_ROWS,BOARD_COLS)),-1)
    play_vs_AI(heuristic_weight, -1)
    # self_play(hp)
    pass