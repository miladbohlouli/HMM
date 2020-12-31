# Hidden Markov Chains

Hidden Markov Chain(HMMs) are a special kind of Markov chains which in contrast to the typical chains the observations are separate from the states. This is why they include hidden in their title. There are three kinds of problems in HMMs:

1. [*Likelihood*](#1-likelihood)
2. [*Decoding*](#2-Decoding)
3. [*Learning*](#3-Learning)


## Into the details of HMMs

Let's discuss some of the basic concept in this area. As you know random variables are some functions from sample space to real numbers, afterwards we encounter the stochastic processes, which are some random variables variant with time. It is notable that Markov processes are a special case of stochastic processes that the future is independent of past knowing the current time. But Markov chains are Markov processes with a countable number of states. Knowing the probability of transitions summarized in a matrix as transition probability matrix among different states is a great step in determining the whole chain. Eventually HMMs are some specific kind of Markov chains that the states and observations are two separate sets. Opposed to Markov chains, Knowing the transition probability matrix is not sufficient for knowing the whole process, we need to know the probability of each observation on any state known as the emission probabilities.

<b>Any HMMs contains a few key components that have to be defined:

$A$: transition probability matrix

$B$ or $P_i(O_t)$: The probability of observation $O_t$ at state i.

$O = o_1, o_2, ..., o_T$: A sequence of observations

$q_1, q_2, ... q_N$: Hidden states

$q_0$: initial state

$q_e$: Final state

</b>

# 1. likelihood
