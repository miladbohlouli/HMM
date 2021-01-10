import numpy as np
import os
from tqdm import tqdm
np.set_printoptions(precision=4, suppress=True)

class hmm:
    def __init__(self, states: list):
        """
        The constructor of the hmm class
        :param states: These are the set of the possible states
        """
        # Converting the states to integers
        states.append("__end__")
        states.insert(0, "__start__")
        self.states_dict = {key: value for value, key in enumerate(states)}
        self.observation_dict = {}
        self.bigram_states_dictionary = {}
        self.unigram_states_dictionary = {}
        self.num_states = len(states)
        self.num_observations = None
        self.__transition_probability = None
        self.__emission_probabilities = None

    def supervised_training(self, x: np.ndarray, y: np.ndarray):
        """
        This is for the case where the states for each individual observation is known,
        using this knowledge the matrixes for transition probabilities and emission probabilities will be
        calculated according to the dataset.
        :param x: The observations in the dataset defined as predictor variables
        :param y: The states in the dataset defined as the labels
        """
        self.observation_dict = {key: value for value, key in enumerate(np.unique(x))}
        self.num_observations = len(self.observation_dict)
        self.__emission_probabilities = np.zeros((self.num_states, self.num_observations))
        self.__transition_probability = np.zeros((self.num_states, self.num_states))
        numerical_x = self.convert_numerical(x, self.observation_dict)
        numerical_y = self.convert_numerical(y, self.states_dict)
        x = []
        y = []

        seq_len = 10
        for i in range(len(numerical_x)-seq_len+1):
            y.append(np.append(np.insert(numerical_y[i:i + seq_len], 0, self.states_dict["__start__"]),
                               self.states_dict["__end__"]))
            x.append(numerical_x[i:i + seq_len])
        x = np.asarray(x)
        y = np.asarray(y)

        # Forming the the bigram dictionary
        for seq in y:
            for i in range(seq_len + 1):
                if (seq[i], seq[i+1]) not in self.bigram_states_dictionary:
                    self.bigram_states_dictionary[(seq[i], seq[i + 1])] = 1
                else:
                    self.bigram_states_dictionary[(seq[i], seq[i + 1])] += 1

        # Learning the transition probability matrix
        for i in range(self.num_states):
            for j in range(self.num_states):
                if (i, j) in self.bigram_states_dictionary:
                    self.__transition_probability[i, j] = self.bigram_states_dictionary[(i, j)] \
                                                          / np.sum(y == i)

        # Learning the observation probability matrixs
        for i in range(self.num_states):
            for j in range(self.num_observations):
                self.__emission_probabilities[i, j] = np.sum(np.logical_and(x == j, y[:, 1:-1] == i)) / np.sum(y == i)

    # Todo:
    def unsupervised_training(self, x: np.ndarray, iterations=10):
        self.observation_dict = {key: value for value, key in enumerate(np.unique(x))}
        self.num_observations = len(self.observation_dict)
        x = self.convert_numerical(x, self.observation_dict)
        seq_len = len(x)

        # Initilizing the two matrixes
        self.__emission_probabilities = np.random.normal(0, 1, (self.num_states, self.num_observations))
        self.__transition_probability = np.random.normal(0, 1, (self.num_states, self.num_states))



        for _ in tqdm(range(iterations)):
            gamma = np.zeros((self.num_states, seq_len))
            zai = np.zeros((self.num_states, self.num_states, seq_len))

            # For computation relaxation construct the backward probability matrix and forward probability matrix
            alpha = self.likelihood()
            zai = np.zeros((self.num_states, seq_len))

            # Expectation step
            for t in range(seq_len):
                



            # Maximization step


    def viterbi(self, observation: list):
        """
        The Viterbi algorithm used for decoding the given observation to the sequence of probable states
        :param observation: The sequence of the observation as a list of the vocabs defined in the dataset
        :return: Returns the set of the states as a list of the provided states.
        """
        num_observation = len(observation)
        numerical_observation = self.convert_numerical(observation, self.observation_dict)

        back_track_matrix = np.zeros((self.num_states, num_observation), dtype=np.int)
        probability_matrix = np.zeros((self.num_states, num_observation))

        # Initilization
        probability_matrix[:, 0] = self.__transition_probability[self.states_dict["__start__"], :] \
                                   * self.__emission_probabilities[:, numerical_observation[0]]

        back_track_matrix[:, 0] = -1

        # Recursion
        for i in range(1, len(observation)):
            for next_state in range(self.num_states):
                temp = np.zeros(self.num_states)
                for initial_state in range(self.num_states):
                    temp[initial_state] = probability_matrix[initial_state, i-1] * \
                                          self.__transition_probability[initial_state, next_state] * \
                                          self.__emission_probabilities[next_state, numerical_observation[i]]

                probability_matrix[next_state, i] = np.max(temp)
                back_track_matrix[next_state, i] = int(np.argmax(temp))

        # Termination and Decoding
        states = np.zeros(len(observation), dtype=np.int)-1
        states[-1] = np.argmax(probability_matrix[:, -1] *
                               self.__transition_probability[:, self.states_dict["__end__"]])
        for i in range(len(observation)-1, 0, -1):
            states[i-1] = back_track_matrix[states[i], i]

        return self.convert_states(states, self.states_dict)

    def likelihood(self, observation: list):
        """
        The likelihood algorithm used for calculating the likelihood of a provided sequence, this algorithm is in
            fact the forward algorithm which will be further used in  unsupervised_learning section.
        :param observation: The sequence of the observation as a list of the vocabs defined in the dataset
        :return: Returns the likelihood of the the provided sequence as a float number and the probability matrix
            which contains the alpha values
        """
        num_observation = len(observation)
        numerical_observation = self.convert_numerical(observation, self.observation_dict)

        probability_matrix = np.zeros((self.num_states, num_observation))

        # Initilization
        probability_matrix[:, 0] = self.__transition_probability[self.states_dict["__start__"], :]\
                                   * self.__emission_probabilities[:, numerical_observation[0]]

        # Recursion
        for i in range(1, len(observation)):
            for next_state in range(self.num_states):
                temp = np.zeros(self.num_states)
                for initial_state in range(self.num_states):
                    temp[initial_state] = probability_matrix[initial_state, i-1] * \
                                          self.__transition_probability[initial_state, next_state] * \
                                          self.__emission_probabilities[next_state, numerical_observation[i]]

                probability_matrix[next_state, i] = np.sum(temp)

        # Termination and calculating the likelihood
        return np.sum(self.__transition_probability[:, self.states_dict["__end__"]] * probability_matrix[:, -1]), \
               probability_matrix

    def backward(self, observation: list):
        numerical_observation = self.convert_numerical(observation, self.observation_dict)

        probability_matrix = np.zeros((self.num_states, len(observation)))

        # Initilization
        probability_matrix[:, -1] = self.__transition_probability[:, self.states_dict["__end__"]]

        # Recursion
        for i in range(len(observation)-2, -1, -1):
            for initial_state in range(self.num_states):
                temp = np.zeros(self.num_states)
                for next_state in range(self.num_states):
                    temp[next_state] = probability_matrix[next_state, i + 1] * \
                                       self.__transition_probability[initial_state, next_state] * \
                                       self.__emission_probabilities[next_state, numerical_observation[i + 1]]

                probability_matrix[initial_state, i] = np.sum(temp)

        # Termination and calculating the backward probability
        backward_likelihood = np.sum((self.__transition_probability[self.states_dict["__start__"], :]) \
                                     * self.__emission_probabilities[:, numerical_observation[0]] \
                                     * probability_matrix[:, 0])

        return backward_likelihood, probability_matrix

    @staticmethod
    def convert_states(numerical_sequence, dictionary):
        """
        Used to convert a numerical numerical sequence to a sequence in terms of states. Since the states and the
        observations have been converted to numerical values, this step is necessary.
        :param numerical_sequence: The provided numerical sequence
        :param dictionary: The dictionary with keys as State values and values as numerical values
        :return: The converted sequence
        """
        result = []
        rev_dec = {value:key for key, value in dictionary.items()}
        for i in range(len(numerical_sequence)):
            result.append(rev_dec[numerical_sequence[i]])
        return result

    @staticmethod
    def convert_numerical(x, dictionary):
        """
        Convert a sequence of observations into numerical values provided as values in dictionary
        :param x: The sequence
        :param dictionary: The dictionary with keys as states or observation values and values as numerical values
        :return: The converted numerical sequence
        """
        return np.array(list(map(lambda x: dictionary[x], x)))


if __name__ == '__main__':
    source_dir = "Dataset/"
    # Read and reform the dataset
    file_read = open(source_dir + [item for item in os.listdir(source_dir) if ".txt" in item][0], "r+")
    lines = file_read.readlines()[1:]
    fix_data = lambda line: line.strip().replace('\n', "").split(",")
    dataset = np.array(list(map(fix_data, lines)), dtype=object)
    states = ["foggy", "rainy", "sunny"]
    my_model = hmm(states)
    # my_model.unsupervised_training(dataset[:, 1])
    my_model.supervised_training(dataset[:, 1], dataset[:, 0])
    print(my_model.likelihood(["no", "yes"]))
    print(my_model.backward(["no", "yes"]))





