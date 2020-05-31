import numpy as np

import connectfour.env.config as config


class ConnectFour:
    DIRECTIONS = {
        'up': (-1, 0),
        'down': (1, 0),
        'right': (0, 1),
        'left': (0, -1),
        'up-left': (-1, -1),
        'up-right': (-1, 1),
        'down-left': (1, -1),
        'down-right': (1, 1)
    }

    PAIRS = [('up', 'down'), ('right', 'left'), ('up-left', 'down-right'), ('up-right', 'down-left')]
    INT_TO_RENDERED = {
        config.INT_P1: config.P1_SYMBOL,
        config.INT_P2: config.P2_SYMBOL,
        config.INT_BLANK: config.BLANK_SYMBOL
    }

    def __init__(self):
        self.grid = [[config.INT_BLANK for j in range(config.COLUMNS)] for i in range(config.ROWS)]
        self.to_move = config.INT_P1
        self.done = False
        self.score = 0

    def step(self, move):
        move_is_valid = self.check_move(move)
        if not move_is_valid:
            raise Exception("{0}\n Invalid move made, {1}".format(self.render(), move))

        p1_is_playing = self.to_move == config.INT_P1
        placed_row = -1
        for i in reversed(range(len(self.grid))):
            if self.grid[i][move] == config.INT_BLANK:
                placed_row = i
                self.grid[i][move] = config.INT_P1 if p1_is_playing else config.INT_P2
                break

        self.done = self.check_if_done(placed_row, move)
        reward = self.score
        done = self.done
        if not self.done:
            self.to_move = config.INT_P2 if p1_is_playing else config.INT_P1
        else:
            self.reset()
        return self.get_state(), reward, done

    def get_state(self):
        return np.array(self.grid.copy()) * self.to_move

    def render(self):
        output = list()
        add = output.append
        add("0 1 2 3 4 5 6\n")
        add("-------------\n")
        for row in self.grid:
            for element in row:
                add(ConnectFour.INT_TO_RENDERED[element])
                add(' ')
            add('\n')
        return ''.join(output)

    def check_move(self, move):
        try:
            cond1 = not 0 <= move < config.COLUMNS
            cond2 = self.grid[0][move] != config.BLANK_SYMBOL
            return cond1 or cond2 or self.done
        except:
            return False

    def get_options(self):
        map = [1, 0, 0]
        options = [map[x] for x in self.grid[0]]
        return options

    def reset(self):
        self.grid = [[config.INT_BLANK for j in range(config.COLUMNS)] for i in range(config.ROWS)]
        self.to_move = config.INT_P1
        self.done = False
        self.score = 0
        return self.get_state()

    def check_if_done(self, row, col):
        grid = self.grid
        to_move = self.to_move

        def check_dir(r, c, direction):
            in_bound = (0 <= r < config.ROWS) and (0 <= c < config.COLUMNS)
            if not in_bound:
                return 0
            if grid[r][c] != to_move:
                return 0
            return 1 + check_dir(r + direction[0], c + direction[1], direction)

        vals = []
        for pair in ConnectFour.PAIRS:
            dir = check_dir(row, col, ConnectFour.DIRECTIONS[pair[0]])
            opp_dir = check_dir(row, col, ConnectFour.DIRECTIONS[pair[1]])
            vals.append(dir + opp_dir - 1)
        longest_row = max(vals)

        if longest_row >= config.NUM_IN_ROW:
            self.score = longest_row
            return True

        for row in grid:
            if config.INT_BLANK in row:
                return False
        self.score = 0
        return True
