from typing import List

import numpy as np

from env.game import ConnectFourState


def tensorize_game_state(game: ConnectFourState):
    repr = np.zeros((5, *game.board.shape))

    repr[0][game.board == ConnectFourState.P1_INT] = 1
    repr[1][game.board == ConnectFourState.P2_INT] = 1
    repr[2][:][:] = int(game.turn == ConnectFourState.P1_INT)
    repr[3][:][:] = int(game.turn == ConnectFourState.P2_INT)

    return repr


def hot_encode_valid_moves(valid_moves_list: List[np.array]) -> np.array:
    ret = np.zeros(len(valid_moves_list, ConnectFourState.NUM_COLS))

    for i in range(len(valid_moves_list)):
        valid_moves = valid_moves_list[i]
        ret[i][valid_moves] = 1
    return ret
