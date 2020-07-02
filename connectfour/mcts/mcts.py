from copy import deepcopy
class MCTSNode:

    def __init__(self, state, evaluator, predictor):
        self.state = state
        self.player = self.state.to_move
        self.reward = 0
        self.visits = 0

        self.parent = None
        self.children = []

    def select(self):

        node = self

        while not node.is_leaf():
            node = node.get_best_uct()


        def select(node):
            if node.is_leaf():
                return node
            return select(node.get_best_uct())

        return select(self)

    def expand(self):

        options = self.state.legal_actions()


        for option in options:
            state_copy = deepcopy(self.state)
            state_copy.step(option)

            child = MCTSNode(state_copy)

            self._add_child(child)

    def evaluate(self):
        state_copy = deepcopy(self.state)

        pass

    def backprop(self, result):
        node = self
        while node is not None:
            node.update(result)
            node = node.parent

    def update(self, result):
        self.reward += result[self.to_move]
        self.visits += 1

    def get_best_uct(self):
        return MCTSNode()

    def _add_child(self, child):
        self.children.append(child)
        child.parent = self


class Evaluator:
