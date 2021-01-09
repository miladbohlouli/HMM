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


## 2. Decoding

## 3. Learning


## What is this project?
In this project a sample dataset is provided, and learning will be accomplished in both a supervised and unsupervised manner. Afterwards given a sequence of observations the most probable cases for hidden states will be estimated(Decoding).
