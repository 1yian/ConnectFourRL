import numpy as np


class ConnectFourState:
    """
    Represents a state within a game of connect four.
    Implements functions for being able to transition to new game states.
    """

    NUM_ROWS = 6
    NUM_COLS = 7

    P1_INT = 1
    P2_INT = -1
    BLANK_INT = 0

    NUM_IN_A_ROW_TO_WIN = 4

    DIRECTIONS = {
        'up': (-1, 0),
        'right': (0, 1),
        'up-right': (-1, 1),
        'down-right': (1, 1)
    }

    def __init__(self):
        self.board = np.zeros((ConnectFourState.NUM_ROWS, ConnectFourState.NUM_COLS))

        self.turn = ConnectFourState.P1_INT

        self.score = 0

        # We store the final_score as an integer inside a list so that the shallow copy doesn't touch it.
        # This means all states will be able to access the final score.
        self._final_score = [0]
        # Same here for final move number.
        self._final_move_number = [0]

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.board == other.board) and (self.turn == other.turn)

    def __repr__(self):
        return "" #TODO

    def __copy__(self):
        game_copy = ConnectFourState()
        game_copy.board = self.board.copy()
        game_copy.turn = self.turn
        game_copy.score = self.score
        return game_copy

    @property
    def final_score(self):
        return self._final_score[0]

    @property
    def final_move_number(self):
        return self._final_move_number[0]

    def get_valid_moves(self):
        """
        Get the legal moves of the game.
        :return: Return a np array of column indices that are available for the player to move in
        """
        top_row = self.board[0]
        temp = np.zeros(top_row.shape)
        temp[top_row == 0] = 1
        return np.nonzero(temp)[0]

    def move(self, column: int):
        """
        Make a move on the ConnectFour board.
        :param column: Which column index (from 0 to num_cols - 1) to place your chip in.
        :return done: Whether or not the game is done following the input move.
        """
        # Checking whether input is valid.
        if self.score != 0:
            raise Exception("Can't make move, game is already done.")
        if type(column) != int:
            raise TypeError("Move must be an integer. Got {}. Input was {}".format(type(column), column))
        if not (ConnectFourState.NUM_COLS > column >= 0):
            raise Exception("Column chosen out of bounds. Input was {}".format(column))
        selected_column = self.board[:, column]
        if all(selected_column != 0):
            raise Exception("Invalid move, column is full. Input was {}".format(column))

        # Find which row the chip is going to be dropped
        correct_row = ConnectFourState.NUM_ROWS - 1
        for element in selected_column[:][:][-1]:
            if element == ConnectFourState.BLANK_INT:
                break
            correct_row -= 1

        # Drop the chip
        self.board[correct_row, column] = self.turn
        done = self._check_done()

        # Swap whose turn it is
        self.turn = ConnectFourState.P1_INT if self.turn == ConnectFourState.P2_INT else ConnectFourState.P2_INT
        self._final_move_number[0] += 1
        return done

    def reset(self):
        """
        Resets the game to the beginning, clear the score to 0.
        :return: Whether the game is done. Always returns false since the game is never done once it begins.
        """
        self.board = np.zeros((ConnectFourState.NUM_ROWS, ConnectFourState.NUM_COLS))
        self.turn = ConnectFourState.P1_INT
        self.score = 0
        return False

    def _check_done(self, row, col):

        def check_dir(r, c, direction):
            # Recursive helper function for finding how many in a row for a certain direction.
            in_bound = (0 <= r < ConnectFourState.NUM_ROWS) and (0 <= c < ConnectFourState.NUM_COLS)
            if not in_bound:
                return 0

            if self.board[r][c] != self.turn:
                return 0
            return 1 + check_dir(r + direction[0], c + direction[1], direction)

        for direction in ConnectFourState.DIRECTIONS.values():
            dir = check_dir(row, col, direction)
            opp_dir = check_dir(row, col, (-direction[0], -direction[1]))
            if dir + opp_dir - 1 >= ConnectFourState.NUM_IN_A_ROW_TO_WIN:
                self.score = 1 if self.turn == ConnectFourState.P1_INT else -1
                self._final_score[0] = self.score
                return True

        for row in self.board:
            if ConnectFourState.BLANK_INT in row:
                return False

        return True

