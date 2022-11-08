from typing import List, Dict
import random
import numpy as np
import sys

random.seed(4)
Actions = ['B', 'C']  # bet/call vs check/fold

class InformationSet():
    def __init__(self):
        self.cumulative_regrets = np.zeros(shape=len(Actions))
        self.strategy_sum = np.zeros(shape=len(Actions))
        self.num_actions = len(Actions)

    def normalize(self, strategy: np.array) -> np.array:
        """Normalize a strategy. If there are no positive regrets,
        use a uniform random strategy"""
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = np.array([1.0 / self.num_actions] * self.num_actions)
        return strategy

    def get_strategy(self, reach_probability: float) -> np.array:
        """Return regret-matching strategy"""
        strategy = np.maximum(0, self.cumulative_regrets)
        strategy = self.normalize(strategy)

        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.array:
        return self.normalize(self.strategy_sum.copy())


class KuhnPoker():
    @staticmethod # Static methods do not require a class instance creation. So, they are not dependent on the state of the object.
    def is_terminal(history: str) -> bool:
        return history in ['BC', 'BB', 'CC', 'CBB', 'CBC']

    @staticmethod
    def get_payoff(history: str, cards: List[str]) -> int:
        """get payoff for 'active' player in terminal history"""
        if history in ['BC', 'CBC']:
            return +1
        else:  # CC or BB or CBB
            payoff = 2 if 'B' in history else 1
            active_player = len(history) % 2
            player_card = cards[active_player]
            opponent_card = cards[(active_player + 1) % 2]
            if player_card == 'K' or opponent_card == 'J':
                return payoff
            else:
                return -payoff


class KuhnCFRTrainer():
    def __init__(self):
        self.infoset_map: Dict[str, InformationSet] = {}

    def get_information_set(self, card_and_history: str) -> InformationSet:
        """add if needed and return"""
        if card_and_history not in self.infoset_map:
            self.infoset_map[card_and_history] = InformationSet()
        return self.infoset_map[card_and_history]

    def cfr(self, cards: List[str], history: str, reach_probabilities: np.array, active_player: int):
        if KuhnPoker.is_terminal(history):
            return KuhnPoker.get_payoff(history, cards)

        my_card = cards[active_player]
        #print('active_player: ', active_player, 'my_card + history: ', my_card + history)
        info_set = self.get_information_set(my_card + history)

        #strategy = info_set.get_strategy(reach_probabilities[active_player])
        if active_player == 0:
            strategy = info_set.get_strategy(reach_probabilities[active_player])
        else:
            strategy = [1, 0]
        #print('active_player: ', active_player, 'strategy: ', strategy)

        opponent = (active_player + 1) % 2
        counterfactual_values = np.zeros(len(Actions))

        for ix, action in enumerate(Actions):
            action_probability = strategy[ix]

            # compute new reach probabilities after this action
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[active_player] *= action_probability

            # recursively call cfr method, next player to act is the opponent
            counterfactual_values[ix] = -self.cfr(cards, history + action, new_reach_probabilities, opponent)
            #print('active_player: ', active_player, 'strategy: ', strategy,'infoset: ', my_card + history, 'cards: ', cards, 'history + action: ', history + action, 'action_probability: ', action_probability, 'new_reach_probs: ', new_reach_probabilities, 'new_reach_probs[active_player]: ', new_reach_probabilities[active_player], )
            print('active_player: ', active_player, 'strategy: ', strategy,'infoset: ', my_card + history, 'cards: ', cards, 'history + action: ', history + action, 'CF_values[ix]: ', counterfactual_values[ix], 'new_reach_probabilities', new_reach_probabilities)
        #print('active_player: ', active_player, 'counterfactual_values: ', counterfactual_values)

        # Value of the current game state is just counterfactual values weighted by action probabilities
        node_value = counterfactual_values.dot(strategy)
        for ix, action in enumerate(Actions):
            #print(reach_probabilities[opponent], counterfactual_values[ix], node_value)
            info_set.cumulative_regrets[ix] += reach_probabilities[opponent] * (counterfactual_values[ix] - node_value)
        ##print('active_player: ', active_player)
        #print('my_card + history: ', my_card + history)
        #print('node_value: ', node_value)
        #print('info_set.cumulative_regrets: ', info_set.cumulative_regrets)
        #print()
        return node_value # counterfactual utility/happiness from being at this game node h

    def train(self, num_iterations: int) -> int:
        util = 0
        kuhn_cards = ['J', 'Q', 'K']
        for iter in range(num_iterations):
            #print('------------------------------------------------------------------')
            #print('iteration: ', iter)
            print()
            cards = random.sample(kuhn_cards, 2)
            history = ''
            reach_probabilities = np.ones(2)
            util += self.cfr(cards, history, reach_probabilities, 0)
        return util

if __name__ == "__main__":
    num_iterations = 200
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    cfr_trainer = KuhnCFRTrainer()
    util = cfr_trainer.train(num_iterations)

    #print(f"\nRunning Kuhn Poker chance sampling CFR for {num_iterations} iterations")
    #print(f"\nExpected average game value (for player 1): {(-1./18):.3f}")
    #print(f"Computed average game value               : {(util / num_iterations):.3f}\n")

    #print("We expect the bet frequency for a Jack to be between 0 and 1/3")
    #print("The bet frequency of a King should be three times the one for a Jack\n")

    #print(f"History  Bet  Pass")
    for name, info_set in sorted(cfr_trainer.infoset_map.items(), key=lambda s: len(s[0])):
        print(f"{name:3}:    {info_set.get_average_strategy()}")

# "C:/Users/Jaime GG-B/AppData/Local/Programs/Python/Python39/python.exe" "c:/Users/Jaime GG-B/Downloads/KuhnPoker_CFR copy.py" 100