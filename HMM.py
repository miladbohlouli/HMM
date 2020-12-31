import numpy as np
from matplotlib import pyplot as plt


class hmm:
    def __init__(self, states: list):

        # Converting the states to integers
        self.states_dict = {key:value for value, key in enumerate(states)}

        pass

    def unsupervised_training(self, x:np.ndarray, y:np.ndarray):
        pass

    def supervised_learning(self, x:np.ndarray):
        pass

    def viterbi(self, observation:list):
        pass

    def make_vocab(self, observations:list):
        pass

if __name__ == '__main__':
    pass





