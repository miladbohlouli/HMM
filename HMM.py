import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

np.set_printoptions(precision=4, suppress=True)

class hmm:
    def __init__(self, initial_distribution:list = None):
        """
        The constructor of the hmm class
        :param initial_distribution: The initial_distribution for the given inputss
        """
        self.states_dict = {}
        self.observation_dict = {}
        self.bigram_states_dictionary = {}
        self.num_states = None
        self.num_observations = None
        self.__transition_probability = None
        self.__emission_probabilities = None
        self.initial_distribution = initial_distribution
        self.__transition_probability_ground_truth = None
        self.__emission_probabilities_ground_truth = None

    def supervised_training(self, x: np.ndarray, y: np.ndarray):
        """
        This is for the case where the states for each individual observation is known,
        using this knowledge the matrixes for transition probabilities and emission probabilities will be
        calculated according to the dataset.
        :param x: The observations in the dataset defined as predictor variables
        :param y: The states in the dataset defined as the labels

        """
        self.__fill_parameters(np.unique(x), np.unique(y))
        self.__emission_probabilities = np.zeros((self.num_states, self.num_observations))
        self.__transition_probability = np.zeros((self.num_states, self.num_states))

        numerical_x = self.__convert_numerical(x, self.observation_dict)
        numerical_y = self.__convert_numerical(y, self.states_dict)
        x = self.__build_sequences(numerical_x)
        y = self.__build_sequences(numerical_y)

        # Forming the the bigram dictionary
        for seq in y:
            for i in range(len(seq)-1):
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
                self.__emission_probabilities[i, j] = np.sum(np.logical_and(x == j, y == i)) / np.sum(y == i)

        self.__transition_probability_ground_truth = self.__transition_probability.copy()
        self.__emission_probabilities_ground_truth = self.__emission_probabilities.copy()

    def unsupervised_training(self, x: np.ndarray, states: list, iterations=3):
        """
        Mainly known as the Baum-Welch algorithm, used for the case where the states for each individual observation
            is not known, thus we will follow a unsupervised approach, meaning with the list of the sequences and the
            list of the states, the transition probabilities and emission probabilities will be calculated.
        :param x: The list of the observations as one sequence
        :param states: The list of the states
        :param iterations: number of the iterations that the model should run
        """
        self.__fill_parameters(np.unique(x), np.array(states))
        self.__emission_probabilities = self.__uniform_probabilty_initilization((self.num_states, self.num_observations))
        self.__transition_probability = self.__uniform_probabilty_initilization((self.num_states, self.num_states))
        sequences = self.__build_sequences(x)
        num_sequences = len(sequences)
        seq_len = len(sequences[0])
        transition_costs = []
        emission_costs = []


        for _ in tqdm(range(iterations)):
            gamma = np.zeros((self.num_states, seq_len, num_sequences))
            zai = np.zeros((self.num_states, self.num_states, seq_len-1, num_sequences))

            # Expectation step
            for seq_num, seq in enumerate(sequences):
                likelihood, alpha = self.likelihood(seq)
                _, beta = self.backward(seq)

                gamma[..., seq_num] = (alpha * beta) / np.sum(alpha * beta, axis=0)

                for i in range(self.num_states):
                    for j in range(self.num_states):
                        for t in range(seq_len-1):
                            zai[i, j, t, seq_num] = alpha[i, t] \
                                                    * self.__transition_probability[i, j] \
                                                    * self.__emission_probabilities[j, self.observation_dict[seq[t]]] \
                                                    * beta[j, t+1]

                zai[..., seq_num] /= likelihood

            # Maximization step
            self.__transition_probability = np.sum(zai, axis=(-1, -2)) / np.sum(zai, axis=(-1, -2, -3))

            for k, v in self.observation_dict.items():
                for j in range(self.num_states):
                    self.__emission_probabilities[j, v] = np.sum(gamma[j,
                                                                       np.where(sequences == k)[1],
                                                                       np.where(sequences == k)[0]])
            self.__emission_probabilities /= np.sum(gamma, axis=(-1, -2))[..., None]

            # Normalizing the calculated probabilities

            if self.__transition_probability_ground_truth is not None:
                transition_costs.append(np.sum((self.__transition_probability_ground_truth - self.__transition_probability)**2))
                emission_costs.append(np.sum((self.__emission_probabilities_ground_truth - self.__emission_probabilities)**2))

        self.__transition_probability /= np.sum(self.__transition_probability, axis=-1)[..., None]
        self.__emission_probabilities /= np.sum(self.__emission_probabilities, axis=-1)[..., None]

        plt.subplot(121)
        plt.plot(list(range(iterations)), transition_costs)
        plt.title("MSE error for transition probabilities")
        plt.subplot(122)
        plt.plot(list(range(iterations)), emission_costs)
        plt.title("MSE error for emission probabilities")
        plt.show()

    def viterbi(self, sequence: list):
        """
        The Viterbi algorithm used for decoding the given observation to the sequence of probable states
        :param sequence: The sequence of the observation as a list of the vocabs defined in the dataset
        :return: Returns the set of the states as a list of the provided states.
        """
        seq_len = len(sequence)
        numerical_observation = self.__convert_numerical(sequence, self.observation_dict)

        back_track_matrix = np.zeros((self.num_states, seq_len), dtype=np.int)
        probability_matrix = np.zeros((self.num_states, seq_len))

        # Initilization
        probability_matrix[:, 0] = self.initial_distribution * self.__emission_probabilities[:, numerical_observation[0]]
        back_track_matrix[:, 0] = -1

        # Recursion
        for i in range(1, len(sequence)):
            for next_state in range(self.num_states):
                temp = np.zeros(self.num_states)
                for initial_state in range(self.num_states):
                    temp[initial_state] = probability_matrix[initial_state, i-1] * \
                                          self.__transition_probability[initial_state, next_state] * \
                                          self.__emission_probabilities[next_state, numerical_observation[i]]

                probability_matrix[next_state, i] = np.max(temp)
                back_track_matrix[next_state, i] = int(np.argmax(temp))

        # Termination and Decoding
        states = np.zeros(len(sequence), dtype=np.int) - 1
        states[-1] = np.argmax(probability_matrix[:, -1])
        for i in range(len(sequence) - 1, 0, -1):
            states[i-1] = back_track_matrix[states[i], i]

        return self.__convert_states(states, self.states_dict)

    def likelihood(self, sequence: list):
        """
        The likelihood algorithm used for calculating the likelihood of a provided sequence, this algorithm is in
            fact the forward algorithm which will be further used in  unsupervised_learning section.
        :param sequence: The sequence of the observation as a list of the vocabs defined in the dataset
        :return: Returns the likelihood of the the provided sequence as a float number and the probability matrix
            which contains the alpha values
        """
        seq_len = len(sequence)
        numerical_observation = self.__convert_numerical(sequence, self.observation_dict)

        probability_matrix = np.zeros((self.num_states, seq_len))

        # Initilization
        probability_matrix[:, 0] = self.initial_distribution * self.__emission_probabilities[:, numerical_observation[0]]

        # Recursion
        for i in range(1, len(sequence)):
            for next_state in range(self.num_states):
                temp = np.zeros(self.num_states)
                for initial_state in range(self.num_states):
                    temp[initial_state] = probability_matrix[initial_state, i-1] * \
                                          self.__transition_probability[initial_state, next_state] * \
                                          self.__emission_probabilities[next_state, numerical_observation[i]]

                probability_matrix[next_state, i] = np.sum(temp)

        # Termination and calculating the likelihood
        return np.sum(probability_matrix[:, -1]), probability_matrix

    def backward(self, sequence: list):
        """
        Backward algorithm which will be used in unsupervised learning (Baum-Welch) algorithm
        :param sequence: The provided sequence which we will calculate the Beta probabilities for.
        :return:
        """
        numerical_observation = self.__convert_numerical(sequence, self.observation_dict)
        probability_matrix = np.zeros((self.num_states, len(sequence)))

        # Initilization
        probability_matrix[:, -1] = 1

        # Recursion
        for i in range(len(sequence) - 2, -1, -1):
            for initial_state in range(self.num_states):
                temp = np.zeros(self.num_states)
                for next_state in range(self.num_states):
                    temp[next_state] = probability_matrix[next_state, i + 1] * \
                                       self.__transition_probability[initial_state, next_state] * \
                                       self.__emission_probabilities[next_state, numerical_observation[i + 1]]

                probability_matrix[initial_state, i] = np.sum(temp)

        # Termination and calculating the backward probability
        backward_likelihood = np.sum((self.initial_distribution
                                     * self.__emission_probabilities[:, numerical_observation[0]]
                                     * probability_matrix[:, 0]))

        return backward_likelihood, probability_matrix

    def __fill_parameters(self, unique_observation: np.ndarray, unique_states: np.ndarray):
        """
        Used for filling the class parameters with the provided unique value for states and observation
            This method is for object oriented purposes
        :param unique_observation: The list of the unique observations
        :param unique_states: The list of the unique states
        """
        self.observation_dict = {key: value for value, key in enumerate(sorted(unique_observation))}
        self.num_observations = len(self.observation_dict)
        self.states_dict = {key: value for value, key in enumerate(sorted(unique_states))}
        self.num_states = len(states)
        if self.initial_distribution is None:
            self.initial_distribution = np.array([1 / self.num_states] * self.num_states)


    @staticmethod
    def __convert_states(numerical_sequence, dictionary):
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
    def __convert_numerical(x, dictionary):
        """
        Convert a sequence of observations into numerical values provided as values in dictionary
        :param x: The sequence
        :param dictionary: The dictionary with keys as states or observation values and values as numerical values
        :return: The converted numerical sequence
        """
        return np.array(list(map(lambda x: dictionary[x], x)))

    @staticmethod
    def __uniform_probabilty_initilization(shape: tuple):
        """
        Given a shape it constructs a matrix with the same shape that its sum along second axis will be 1
        :param shape: The shape of the matrix
        :return: The probability matrix
        """
        mat = np.random.uniform(0, 1, shape)
        return mat / np.sum(mat, axis=1)[..., None]

    @staticmethod
    def __build_sequences(x, seq_len: int = 10):
        """
        Used to convert the given one sequence to sequences with specific length using a sliding window
        :param x: The provided input dataset
        :param seq_len: The length of the desired sequences
        :return: np.ndarray containing the sequences
        """
        sequences = []
        for i in range(len(x) - seq_len + 1):
            sequences.append(x[i:i + seq_len])
        return  np.asarray(sequences)

    def __str__(self):
        return f"{self.__transition_probability} \n {self.__emission_probabilities}"


if __name__ == '__main__':
    source_dir = "Dataset/"
    # Read and reform the dataset
    file_read = open(source_dir + [item for item in os.listdir(source_dir) if ".txt" in item][0], "r+")
    lines = file_read.readlines()[1:]
    fix_data = lambda line: line.strip().replace('\n', "").split(",")
    dataset = np.array(list(map(fix_data, lines)), dtype=object)
    states = ["foggy", "rainy", "sunny"]
    my_model = hmm()
    my_model.supervised_training(dataset[:, 1], dataset[:, 0])
    print(my_model)

    my_model.unsupervised_training(dataset[:, 1], states, iterations=40)
    print(my_model)





    # print(my_model.likelihood(["no"])[0])
    # print(my_model.backward(["yes"])[0])
    # print(my_model.viterbi(["no", "no", "no", "no"]))
    # print(my_model.backward(["no", "yes"]))





