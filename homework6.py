
"""
    Given observations, find hidden state sequence by viterbi algorithm
"""
import numpy as np
from hmmlearn import hmm

states = ["B1", "B2", "B3"]
n_states = len(states)   # =3

observations = ["red","red","yellow","green","yellow"]
n_observations = len(observations)

start_probability = np.array([0.4, 0.35, 0.25])

transition_probability = np.array([
  [0.3, 0.2, 0.5],
  [0.1, 0.3, 0.6],
  [0.7, 0.25, 0.05]
])

emission_probability = np.array([
  [0.8,0.1,0.1],
  [0.2,0.4,0.4],
  [0.15,0.25,0.6]
])

model = hmm.MultinomialHMM(n_components=n_states)
# MultinomialHMM: observation distribution in Multinomial
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

seen = np.array([[0,0,2,1,2]]).T        # 0: red;     1: green; 2:yellow   => r g y
logprob, box = model.decode(seen, algorithm="viterbi")
seen = [0,0,2,1,2]
print("The m&m candy picked:", ", ".join(map(lambda x: observations[x], seen)))
print("The hidden box:", ", ".join(map(lambda x: states[x], box)))

"""
    Find probability of observation sequence
"""
seen = np.array([[0,0,2,1,2]]).T                  # P(rwr) =?
p_rrygy=seen
print(f"The probability of observation sequence is red-red-yellow-green-yellow is  { model.score(seen)}")
