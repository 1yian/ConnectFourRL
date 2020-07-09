from copy import deepcopy
import math, sys
import random
import time

class MCTSNode:
    C_PUCT = 4

    def __init__(self, state, evaluator, predictor, existing_nodes={}):

        self.priors = None
        self.eval = None

        self.evaluator, self.predictor = evaluator, predictor
        self.existing_nodes = existing_nodes

        self.state = state
        self.player = self.state.to_move

        self.reward, self.visits, self.q = 0, 0, 0

        self.parents = []
        self.children = {}

    def select(self):
        node = self
        while not node._is_leaf():
            node = node.get_best_puct()

        return node

    def expand(self):

        options = self.state.legal_actions()
        for option in options:
            state_copy = deepcopy(self.state)
            state_copy.step(option)
            self._add_child(state_copy, option)

    def mcts(self, iters, greedy=True):
        t = time.time()
        while time.time() - t < 15:
            leaf = self.select()
            leaf.expand()
            result = leaf.select().evaluate()
            leaf.backprop(result)

        max_visits = -1
        best_node = None
        print([child.visits for child in self.children])
        print([child.q for child in self.children])

        for child in self.children:
            if max_visits < child.visits:
                best_node = child
                max_visits = child.visits
        return best_node, self.children[best_node]

    def evaluate(self):
        return self.evaluator.evaluate(self)

    def backprop(self, result):
        def backprop_recursive(node, result):
            node.update(result)
            for parent in node.parents:
                backprop_recursive(parent, result)

        backprop_recursive(self, result)

    def update(self, result):
        self.reward += result[self.player]
        self.visits += 1
        self.q = self.reward / self.visits

    def get_best_puct(self):
        if self.visits == 0:
            return random.choice(list(self.children.keys()))

        prediction = self.predictor.predict(self)
        sqrt_total_child_visits = math.sqrt(sum([child.visits for child in self.children]))

        max_puct = -99999
        max_puct_node = None
        tmp = list(self.children.keys())
        random.shuffle(tmp)
        for child in tmp:

            action = self.children[child]
            prior = prediction[action]
            q = child.q
            u = MCTSNode.C_PUCT * prior * sqrt_total_child_visits / (1 + child.visits)
            puct_val = q + u
            if max_puct < puct_val:
                max_puct = puct_val
                max_puct_node = child
        return max_puct_node

    def _add_child(self, state, action):
        state_exists = False
        for node_state in self.existing_nodes:
            if state == node_state:
                state_exists = True
        if state_exists:
            child = self.existing_nodes[state]
        else:
            child = MCTSNode(state, self.evaluator, self.predictor, self.existing_nodes)
        self.children[child] = action
        child.parents.append(self)

    def _is_leaf(self):
        return len(self.children) == 0


class RandomPredictor:
    def predict(self, state):
        num_actions = len(state.legal_actions())
        return [1 / num_actions] * 7


class RandomEvaluator:
    def evaluate(self, state):
        state_copy = deepcopy(state)
        while not state_copy.done:
            state_copy.step(random.choice(state_copy.legal_actions()))
        score = abs(state_copy.score)
        result = {state_copy.to_move: -score, -state_copy.to_move: score}
        return result


from connectfour.a2c.agent import ActorCriticAgent


class A2C:
    def __init__(self):
        self.priors = {}
        self.vals = {}
        agent = ActorCriticAgent()
        agent.load_checkpoint('./checkpoints/a2c_240.pt')
        self.agent = agent

    def predict(self, s):
        if s.priors is not None:
            return s.priors
        state = s.state.get_state()
        priors, _ = self.agent.get_prediction([state])
        s.priors = priors[0].cpu()
        return s.priors

    def evaluate(self, s):
        if s.eval is not None:
            return s.eval
        if s.state.done:
            score = abs(s.state.score)
            result = {s.state.to_move: -score, -s.state.to_move: score}
            return result
        state = s.state.get_state()
        _, vals = self.agent.get_prediction([state])
        score = vals[0].cpu()
        result = {s.state.to_move: -score, -s.state.to_move: score}
        s.eval = result
        return result
