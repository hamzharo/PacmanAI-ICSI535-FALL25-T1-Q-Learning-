# qlearningAgents.py
# ------------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random, util, math
import numpy as np

##########################################################
# CLASS: QLearningAgent
##########################################################

class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    This agent learns an action-value function Q(s, a) that estimates how
    good it is to take action 'a' in state 's' and follow the best policy
    thereafter.

    Key parameters inherited from ReinforcementAgent:
      - self.epsilon: exploration probability (ε)
      - self.alpha: learning rate (α)
      - self.discount: discount factor (γ)
    
    Methods to Implement:
      - getQValue
      - computeValueFromQValues
      - computeActionFromQValues
      - getAction
      - update
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # Using util.Counter() -> default value 0.0 for unseen keys
        self.qvalues = util.Counter()
        print("[INIT] Q-Learning Agent initialized.")

    ##################################################
    # Return current Q-value for (state, action)
    ##################################################
    def getQValue(self, state, action):
        """
        Returns Q(state, action)
        If (state, action) has never been seen before, returns 0.0
        """
        value = self.qvalues[(state, action)]
        print(f"[GET_Q] State={state}, Action={action}, Q={value:.4f}")
        return value

    ##################################################
    # Compute the best Q-value for a given state
    ##################################################
    def computeValueFromQValues(self, state):
        """
        Returns max_a Q(state, a)
        If there are no legal actions (terminal state), returns 0.0
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            print(f"[COMPUTE_VALUE] Terminal state={state}, returning 0.0")
            return 0.0

        maxQ = max([self.getQValue(state, a) for a in legalActions])
        print(f"[COMPUTE_VALUE] State={state}, MaxQ={maxQ:.4f}")
        return maxQ

    ##################################################
    # Compute the best action to take in a state
    ##################################################
    def computeActionFromQValues(self, state):
        """
        Returns the best action according to current Q-values.
        If no legal actions, returns None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            print(f"[DEBUG] No legal actions available in state: {state}")
            return None
        else:
            print(f"[DEBUG] Legal actions in state: {legalActions}")


        # Compute all actions with max Q-value (to break ties randomly)
        maxQ = self.computeValueFromQValues(state)
        bestActions = [a for a in legalActions if self.getQValue(state, a) == maxQ]

        # Choose randomly among the best
        bestAction = random.choice(bestActions)
        print(f"[COMPUTE_ACTION] State={state}, BestAction={bestAction}, Q={maxQ:.4f}")
        return bestAction

    ##################################################
    # Choose action using epsilon-greedy strategy
    ##################################################
    def getAction(self, state):
        """
        With probability ε -> choose random action (exploration)
        With probability (1-ε) -> choose best known action (exploitation)
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Exploration decision
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
            print(f"[ACTION] Random exploration: {action}")
        else:
            action = self.computeActionFromQValues(state)
            print(f"[ACTION] Greedy exploitation: {action}")

        return action

    ##################################################
    # Q-learning update rule
    ##################################################
    def update(self, state, action, nextState, reward: float):
        """
        Performs the Q-learning update for a state transition:
        
        Q(s,a) ← (1 - α) * Q(s,a) + α * [r + γ * max_a' Q(s', a')]
        """
        oldQ = self.getQValue(state, action)
        nextValue = self.computeValueFromQValues(nextState)
        sample = reward + self.discount * nextValue

        # Update rule
        newQ = (1 - self.alpha) * oldQ + self.alpha * sample
        self.qvalues[(state, action)] = newQ

        print(f"[UPDATE] State={state}, Action={action}, Reward={reward}, NextValue={nextValue:.4f}")
        print(f"         OldQ={oldQ:.4f}, NewQ={newQ:.4f}")

    ##################################################
    # Policy and Value accessors
    ##################################################
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


##########################################################
# CLASS: PacmanQAgent
##########################################################
class PacmanQAgent(QLearningAgent):
    """
    Same as QLearningAgent, but with Pac-Man-specific defaults:
      - epsilon = 0.05
      - gamma = 0.8
      - alpha = 0.2
    """

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # Pacman index
        QLearningAgent.__init__(self, **args)
        print(f"[INIT] PacmanQAgent initialized with ε={epsilon}, α={alpha}, γ={gamma}")

    def getAction(self, state):
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


##########################################################
# CLASS: ApproximateQAgent
##########################################################
class ApproximateQAgent(PacmanQAgent):
    """
    Approximate Q-Learning Agent:
    Instead of using a Q-table for (state, action), this agent
    learns weights for features extracted from (state, action).
    Q(s,a) = w · f(s,a)
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        print("[INIT] ApproximateQAgent initialized with feature extractor:", extractor)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Q(s,a) = Σ (w_i * f_i(s,a))
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = self.weights * features  # util.Counter supports dot product
        print(f"[APPROX_Q] State={state}, Action={action}, Q={q_value:.4f}")
        return q_value

    def update(self, state, action, nextState, reward: float):
        """
        Update weights using the gradient descent rule:
        difference = [r + γ * max_a' Q(s',a')] - Q(s,a)
        w_i ← w_i + α * difference * f_i(s,a)
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = self.getQValue(state, action)
        nextValue = self.computeValueFromQValues(nextState)
        difference = (reward + self.discount * nextValue) - q_value

        for f in features:
            self.weights[f] += self.alpha * difference * features[f]

        print(f"[APPROX_UPDATE] State={state}, Action={action}, Reward={reward}")
        print(f"                Difference={difference:.4f}, Weights={dict(self.weights)}")

    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            print("[TRAINING COMPLETE] Final learned weights:")
            print(dict(self.weights))
