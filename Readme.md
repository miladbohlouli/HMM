# Hidden Markov Chains

Hidden Markov Chain(HMMs) are a special kind of Markov chains which in contrast to the typical chains the observations are separate from the states. This is why they include hidden in their title. There are three kinds of problems in HMMs:

1. [*Likelihood*](#1-likelihood)
2. [*Decoding*](#2-Decoding)
3. [*Learning*](#3-Learning)


## Into the details of HMMs

Let's discuss some of the basic concept in this area. As you know random variables are some functions from sample space to real numbers, afterwards we encounter the stochastic processes, which are some random variables variant with time. It is notable that Markov processes are a special case of stochastic processes that the future is independent of past knowing the current time. But Markov chains are Markov processes with a countable number of states. Knowing the probability of transitions summarized in a matrix as transition probability matrix among different states is a great step in determining the whole chain. Eventually HMMs are some specific kind of Markov chains that the states and observations are two separate sets. Opposed to Markov chains, Knowing the transition probability matrix is not sufficient for knowing the whole process, we need to know the probability of each observation on any state known as the emission probabilities.

<b>Any HMMs contains a few key components that have to be defined:

A or P_i(O_t): transition probability matrix

B or emission probabilities: The probability of observation $O_t$ at state i.

O = o_1, o_2, ..., o_T: A sequence of observations

q_1, q_2, ... q_N: Hidden states

q_0: initial state

q_e: Final state

</b>

## 1. likelihood
This is for the case that we are given a sequence of observation (O) and we would like to know the likelihood (P(O)) of this observation, assuming that we have learnt the hmm and we know both the matrixes A and B.

If we knew the states of each observation time-step, then the calculation of likelihood would be as easy as pie. To better understand this, let's see an example (This is a well known example in this field).

> Example: Imagine for the weather on each day we have two cases, but we don't have any clue what they mean, correspondingly we have some other observations that we are dealing with, e.g. the number of ice creams that have been eaten on that specific day. In this case a possible sequence of observation could be 3, 1, 3 meaning that on three consecutive days the number of eaten ice creams have been 3 ,1 ,3 respectively. what we are interested in Likelihood problem is to calculate the probability of this sequence as P(3 1 3). If we knew the states behind the states it would be a lot easier for us to calculate the likelihood as p(3 1 3 | hot cold hot) = p(3|hot)p(1|cold)p(3|hot). Using this approach we could have calculated the joint probability P(3 1 3, hot cold hot) and after all the probability of observation would be the marginal probability of the joint probability. So we should have added all of the cases that we may have for the states. Since we have a sequence of three time-steps and 2 cases for each time-step, then we can conclude that the total number of cased is 2^3 = 8. This is a problem since the growth in length will increase the number of the cases exponentially. To solve this we will use the forward algorithm.

With a little help of math and probability, we will calculate the probabilities of reaching a specific sequence with all possible cases for hidden values, which are called alpha values. Eventually the alpha values, will be sum of all the possible sequences of hidden states.


## 2. Decoding
Given a sequence of observations, the goal is to estimate the probable hidden states behind these observations. The method used here is similar to likelihood described in section 1, but with the difference that instead of calculating the sum, we will obtain the maximum of the calculated values recursively. Additionally we should keep track of states that have had the maximum value on every time-step. Eventually the most probable sequence of states will be estimated using values calculated for the final observation in the sequence. This algorithm is commonly knowns as the <b> Viterbi algorithm.</b>

## 3. Learning
Preceding Decoding and likelihood, we should learn the previously discussed matrixes, transition probability matrix and emission probability matrix. There are two main approaches for learning, which we will consider in the next parts.

### 3.1 Supervised learning
Given some sequences of observations, and their corresponding hidden states, using the conditional probability rules the transition probability matrix and the emission probability matrix will be calculated. This is a straight-forward algorithm, but most of the times, the only thing we know is the unique possible states, we have no additional knowledge about the states behind each observation. This is where we will utilize the expectation maximization approach to estimate the discussed matrixes.

### 3.2 Unsupervised learning known as Baum-Welch algorithm
In contrast with the previous section, we are given only the list of the observations and possible states. Using the Expectation Maximization approach we will try to estimate the parameters of the model iteratively.


## What is this project?
Given a dataset containing the observations as if a person has taken umbrella going out and the weather condition of that specific day being foggy, sunny or rainy as the hidden state, all of the previously discussed algorithms have been implemented. The further details of the implementations are included inside the implementation files. A jupyter notebook file is attached with some possible test cases in both the unsupervised and supervised learning cases.
