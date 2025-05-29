import random

# 1. Q-Learning
class QLearningAgent:
    def __init__(self, game, discount, learning_rate, explore_prob):
        self.game = game
        self.discount = discount
        self.learning_rate = learning_rate
        self.explore_prob = explore_prob
        self.q_table = {}
        self.visit_count = {}

    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        return self.q_table.get((state, action), 0)


    def get_value(self, state):
        """Compute state value from Q-value.
        V(s) = max_a Q(s,a)
        """
        actions = self.game.get_actions(state)
        if not actions:
            return 0
        return max(self.get_q_value(state, a) for a in actions)

    def f(self, state, action):
        visit_count = self.visit_count.get((state, action), 0)
        bonus = 1.5
        return self.get_q_value(state, action) + bonus / (visit_count + 1)

    def get_best_policy(self, state):
        """Compute the best action to take in the state using Policy Extraction.
        π(s) = argmax_a Q(s,a)
        """
        max_q = float("-inf")
        for a in self.game.get_actions(state):
            curr_q = self.f(state, a)
            if max_q < curr_q:
                max_q = curr_q

        best_actions = [a for a in self.game.get_actions(state) if self.f(state, a) == max_q]
        return random.choice(best_actions)

    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.
        """
        q_value = (1 - self.learning_rate)* self.get_q_value(state, action) + self.learning_rate * (reward + self.discount * self.get_value(next_state))
        print(q_value, self.get_value(next_state), state, action)
        self.q_table[(state, action)] = q_value

        # f
        if (state, action) not in self.visit_count:
            self.visit_count[(state, action)] = 0
        self.visit_count[(state, action)] += 1
        return q_value

    # 2. Epsilon Greedy
    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.
        """
        actions = self.game.get_actions(state)
        if not actions:
            return None
        if random.random() < self.explore_prob:
            return random.choice(list(actions))
        else:
            return self.get_best_policy(state)

# 4. Approximate Q-Learning
class ApproximateQAgent(QLearningAgent):
    def __init__(self, *args, extractor):
        super().__init__(*args)
        self.extractor = extractor
        self.weight = {}

    def get_weight(self, feature):
        """Get weight of a feature.
        Never seen feature should have a weight of 0.
        """
        return self.weight.get(feature, 0)

    def get_q_value(self, state, action):
        """Compute Q value based on the dot product of feature components and weights.
        Q(s,a) = w_1 * f_1(s,a) + w_2 * f_2(s,a) + ... + w_n * f_n(s,a)
        """
        feature_dict = self.extractor(state, action)
        Q = 0
        for key, value in feature_dict.items():
            Q += self.get_weight(key) * value
        return Q

    def update(self, state, action, next_state, reward):
        """Update weights using TD.
        Δ = R + γ V(s') - Q(s,a)
        Then update weights: w_i = w_i + α * Δ * f_i(s, a)
        """
        feature_dict = self.extractor(state, action)
        TD = reward + self.discount * self.get_value(next_state) - self.get_q_value(state, action)
        for feature, f_value in feature_dict.items():
            new_w = self.get_weight(feature) + self.learning_rate * TD * f_value
            self.weight[feature] = new_w

