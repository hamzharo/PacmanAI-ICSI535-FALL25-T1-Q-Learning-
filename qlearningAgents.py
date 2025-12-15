# # qlearningAgents.py
# # ------------------
# # Licensing Information: You are free to use or extend these projects for
# # educational purposes provided that (1) you do not distribute or publish
# # solutions, (2) you retain this notice, and (3) you provide clear
# # attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# #
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# # The core projects and autograders were primarily created by John DeNero
# # (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # Student side autograding was added by Brad Miller, Nick Hay, and
# # Pieter Abbeel (pabbeel@cs.berkeley.edu).

# from game import *
# from learningAgents import ReinforcementAgent
# from featureExtractors import *
# import random, util, math
# import numpy as np

# ##########################################################
# # CLASS: QLearningAgent
# ##########################################################

# class QLearningAgent(ReinforcementAgent):
#     """
#     Q-Learning Agent

#     This agent learns an action-value function Q(s, a) that estimates how
#     good it is to take action 'a' in state 's' and follow the best policy
#     thereafter.

#     Key parameters inherited from ReinforcementAgent:
#       - self.epsilon: exploration probability (ε)
#       - self.alpha: learning rate (α)
#       - self.discount: discount factor (γ)
    
#     Methods to Implement:
#       - getQValue
#       - computeValueFromQValues
#       - computeActionFromQValues
#       - getAction
#       - update
#     """

#     def __init__(self, **args):
#         "You can initialize Q-values here..."
#         ReinforcementAgent.__init__(self, **args)
#         # Using util.Counter() -> default value 0.0 for unseen keys
#         self.qvalues = util.Counter()
#         print("[INIT] Q-Learning Agent initialized.")

#         ### 🔹 Add training record tracking
#         self.episode_rewards = []          # store total reward per episode
#         self.q_updates = 0                 # count number of Q-value changes in this episode
#         self.log_file = "training_log.txt" # log file name

#         # Create / reset the log file
#         with open(self.log_file, "w") as f:
#             f.write("Episode,TotalReward,UpdatedQValues\n")

#     ##################################################
#     # Return current Q-value for (state, action)
#     ##################################################
#     def getQValue(self, state, action):
#         """
#         Returns Q(state, action)
#         If (state, action) has never been seen before, returns 0.0
#         """
#         key = (state, action)
#           # 🔑 use hash(state) instead of raw object
#         value = self.qvalues[key]
#         print(f"[GET_Q] Pos={state}, Action={action}, Q={value:.4f}")

#         return value


#     ##################################################
#     # Compute the best Q-value for a given state
#     ##################################################
#     def computeValueFromQValues(self, state):
#         """
#         Returns max_a Q(state, a)
#         If there are no legal actions (terminal state), returns 0.0
#         """
#         legalActions = self.getLegalActions(state)
#         if not legalActions:
#             print(f"[COMPUTE_VALUE] Terminal state={state}, returning 0.0")
#             return 0.0

#         maxQ = max([self.getQValue(state, a) for a in legalActions])
#         print(f"[COMPUTE_VALUE] State={state}, MaxQ={maxQ:.4f}")
#         return maxQ

#     ##################################################
#     # Compute the best action to take in a state
#     ##################################################
#     def computeActionFromQValues(self, state):
#         """
#         Returns the best action according to current Q-values.
#         If no legal actions, returns None.
#         """
#         legalActions = self.getLegalActions(state)
#         if not legalActions:
#             print(f"[DEBUG] No legal actions available in state: {state}")
#             return None
#         else:
#             print(f"[DEBUG] Legal actions in state: {legalActions}")

#         # Compute all actions with max Q-value (to break ties randomly)
#         maxQ = self.computeValueFromQValues(state)
#         bestActions = [a for a in legalActions if self.getQValue(state, a) == maxQ]

#         # Choose randomly among the best
#         bestAction = random.choice(bestActions)
#         print(f"[COMPUTE_ACTION] State={state}, BestAction={bestAction}, Q={maxQ:.4f}")
#         return bestAction

#     ##################################################
#     # Choose action using epsilon-greedy strategy
#     ##################################################
#     def getAction(self, state):
#         """
#         With probability ε -> choose random action (exploration)
#         With probability (1-ε) -> choose best known action (exploitation)
#         """
#         legalActions = self.getLegalActions(state)
#         if not legalActions:
#             return None

#         # Exploration decision
#         if util.flipCoin(self.epsilon):
#             action = random.choice(legalActions)
#             print(f"[ACTION] Random exploration: {action}")
#         else:
#             action = self.computeActionFromQValues(state)
#             print(f"[ACTION] Greedy exploitation: {action}")

#         return action

#     ##################################################
#     # Q-learning update rule (stable version)
#     ##################################################
#     def update(self, state, action, nextState, reward: float):
#         """
#         Performs the Q-learning update for a state transition:
        
#         Q(s,a) ← (1 - α) * Q(s,a) + α * [r + γ * max_a' Q(s', a')]
#         """
#         # Use stable key representation (position + food layout)
#         key = (state,  action)
#         nextKeyActions = self.getLegalActions(nextState)

#         # Retrieve old Q
#         oldQ = self.qvalues[key]

#         # Compute target
#         nextValue = 0.0
#         if nextKeyActions:
#             nextValue = max([self.getQValue(nextState, a) for a in nextKeyActions])
#         target = reward + self.discount * nextValue

#         # Q-learning update
#         newQ = (1 - self.alpha) * oldQ + self.alpha * target
#         self.qvalues[key] = newQ

#         # Count updates only when value changes
#         if abs(newQ - oldQ) > 1e-6:
#             self.q_updates += 1

#         print(f"[UPDATE] StatePos={state}, Action={action}, Reward={reward:.2f}, NextValue={nextValue:.4f}")
#         print(f"         OldQ={oldQ:.4f}, NewQ={newQ:.4f}")

#     ##################################################
#     # Policy and Value accessors
#     ##################################################
#     def getPolicy(self, state):
#         return self.computeActionFromQValues(state)

#     def getValue(self, state):
#         return self.computeValueFromQValues(state)

#     ##################################################
#     # 🔹 Logging at the end of each episode
#     ##################################################
#     def final(self, state):
#         """
#         Called by the environment at the end of each episode.
#         Logs total reward and how many Q-values changed.
#         """
#         try:
#             ReinforcementAgent.final(self, state)
#         except Exception:
#             pass

#         total_reward = 0
#         if hasattr(self, "episodeRewards") and isinstance(self.episodeRewards, (list, tuple)) and len(self.episodeRewards) > 0:
#             total_reward = self.episodeRewards[-1]
#         elif hasattr(self, "episodeRewards"):
#             total_reward = float(self.episodeRewards)
#         elif hasattr(self, "episode_rewards") and len(self.episode_rewards) > 0:
#             total_reward = self.episode_rewards[-1]

#         if not hasattr(self, "episode_rewards"):
#             self.episode_rewards = []
#         self.episode_rewards.append(total_reward)

#         try:
#             with open(self.log_file, "a") as f:
#                 f.write(f"{getattr(self, 'episodesSoFar', 0)},{total_reward:.2f},{self.q_updates}\n")
#         except Exception as e:
#             print(f"[LOGGING ERROR] {e}")

#         print(f"[EPISODE END] #{getattr(self, 'episodesSoFar', 0)}  TotalReward={total_reward:.2f}, QUpdates={self.q_updates}")
#         self.q_updates = 0



# ##########################################################
# # CLASS: PacmanQAgent
# ##########################################################
# class PacmanQAgent(QLearningAgent):
#     """
#     Same as QLearningAgent, but with Pac-Man-specific defaults:
#       - epsilon = 0.05
#       - gamma = 0.8
#       - alpha = 0.2
#     """

#     def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
#         args['epsilon'] = epsilon
#         args['gamma'] = gamma
#         args['alpha'] = alpha
#         args['numTraining'] = numTraining
#         self.index = 0  # Pacman index
#         QLearningAgent.__init__(self, **args)
#         print(f"[INIT] PacmanQAgent initialized with ε={epsilon}, α={alpha}, γ={gamma}")

#     def getAction(self, state):
#         action = QLearningAgent.getAction(self, state)
#         self.doAction(state, action)
#         return action

#     def final(self, state):
#         """
#         Ensure PacmanQAgent also logs and ends episode properly.
#         """
#         QLearningAgent.final(self, state)


# ##########################################################
# # CLASS: ApproximateQAgent
# ##########################################################
# class ApproximateQAgent(PacmanQAgent):
#     """
#     Approximate Q-Learning Agent:
#     Instead of using a Q-table for (state, action), this agent
#     learns weights for features extracted from (state, action).
#     Q(s,a) = w · f(s,a)
#     """

#     def __init__(self, extractor='IdentityExtractor', **args):
#         self.featExtractor = util.lookup(extractor, globals())()
#         PacmanQAgent.__init__(self, **args)
#         self.weights = util.Counter()
#         print("[INIT] ApproximateQAgent initialized with feature extractor:", extractor)

#     def getWeights(self):
#         return self.weights

#     def getQValue(self, state, action):
#         """
#         Q(s,a) = Σ (w_i * f_i(s,a))
#         """
#         features = self.featExtractor.getFeatures(state, action)
#         q_value = 0.0
#         for feature, value in features.items():
#             q_value += self.weights[feature] * value
#         print(f"[GET_Q] Pos={state.getPacmanPosition()}, Action={action}, Q={q_value:.4f}")
#         return q_value

#     def update(self, state, action, nextState, reward: float):
#         """
#         Performs the approximate Q-learning weight update for a state transition:
#         difference = (r + γ * max_a' Q(s', a')) - Q(s, a)
#         w_i ← w_i + α * difference * f_i(s, a)
#         """
#         # Extract features and current Q-value
#         features = self.featExtractor.getFeatures(state, action)
#         currentQ = self.getQValue(state, action)

#         # Compute target and difference
#         nextValue = self.computeValueFromQValues(nextState)
#         sample = reward + self.discount * nextValue
#         difference = sample - currentQ

#         # Update each weight
#         for feature, value in features.items():
#             self.weights[feature] += self.alpha * difference * value

#         print(f"[APPROX_UPDATE] State={state}, Action={action}, Reward={reward}, NextValue={nextValue:.4f}")
#         print(f"                CurrentQ={currentQ:.4f}, Difference={difference:.4f}, Weights={dict(self.weights)}")

#     def final(self, state):
#         PacmanQAgent.final(self, state)
#         if self.episodesSoFar == self.numTraining:
#             print("[TRAINING COMPLETE] Final learned weights:")
#             print(dict(self.weights))







# qlearningAgents.py
# ------------------

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random, util, math
import numpy as np


############################################################
# QLearningAgent
############################################################

class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent supporting:
      - epsilonGreedy (default)
      - greedy
      - softmax
    """

    def __init__(self, **args):

        # Remove custom fields BEFORE passing args upward
        self.policy = args.pop('policy', 'epsilonGreedy')
        self.temperature = float(args.pop('temperature', 1.0))

        # Parent initialization (safe now)
        ReinforcementAgent.__init__(self, **args)

        # Q-table
        self.qvalues = util.Counter()

        # Logging
        self.q_updates = 0
        self.episode_rewards = []
        self.log_file = "training_log.txt"

        with open(self.log_file, "w") as f:
            f.write("Episode,Reward,QUpdates\n")

        print(f"[INIT] Q-Learning Agent initialized with policy: {self.policy}")


    ############################################################
    # Q-Value helpers
    ############################################################

    def getQValue(self, state, action):
        return self.qvalues[(state, action)]

    def computeValueFromQValues(self, state):
        legal = self.getLegalActions(state)
        if not legal:
            return 0.0
        return max(self.getQValue(state, a) for a in legal)

    def computeActionFromQValues(self, state):
        legal = self.getLegalActions(state)
        if not legal:
            return None

        maxQ = self.computeValueFromQValues(state)
        best = [a for a in legal if self.getQValue(state, a) == maxQ]
        return random.choice(best)


    ############################################################
    # Softmax helper
    ############################################################

    def _softmaxAction(self, state, legal):
        tau = max(self.temperature, 1e-6)
        q_vals = [self.getQValue(state, a) for a in legal]
        max_q = max(q_vals)

        exp_vals = [math.exp((q - max_q) / tau) for q in q_vals]
        total = sum(exp_vals)
        if total == 0:
            return random.choice(legal)

        probs = [ev / total for ev in exp_vals]

        r = random.random()
        cum = 0
        for action, p in zip(legal, probs):
            cum += p
            if r < cum:
                return action

        return legal[-1]


    ############################################################
    # Action selection
    ############################################################

    def getAction(self, state):

        legal = self.getLegalActions(state)
        if not legal:
            return None

        if self.policy == "epsilonGreedy":
            if util.flipCoin(self.epsilon):
                return random.choice(legal)
            return self.computeActionFromQValues(state)

        if self.policy == "greedy":
            return self.computeActionFromQValues(state)

        if self.policy == "softmax":
            return self._softmaxAction(state, legal)

        # Default fallback
        if util.flipCoin(self.epsilon):
            return random.choice(legal)
        return self.computeActionFromQValues(state)


    ############################################################
    # Q-learning update
    ############################################################

    def update(self, state, action, nextState, reward):

        oldQ = self.qvalues[(state, action)]
        legalNext = self.getLegalActions(nextState)
        nextVal = max((self.getQValue(nextState, a) for a in legalNext), default=0.0)

        target = reward + self.discount * nextVal
        newQ = (1 - self.alpha) * oldQ + self.alpha * target

        self.qvalues[(state, action)] = newQ

        if abs(newQ - oldQ) > 1e-6:
            self.q_updates += 1


    ############################################################
    # Safe final()
    ############################################################

    def final(self, state):

        try:
            ReinforcementAgent.final(self, state)
        except:
            pass

        # episodeRewards is a FLOAT
        total = float(self.episodeRewards) if hasattr(self, "episodeRewards") else 0.0

        if not hasattr(self, "episode_rewards"):
            self.episode_rewards = []
        self.episode_rewards.append(total)

        with open(self.log_file, "a") as f:
            f.write(f"{getattr(self,'episodesSoFar',0)},{total},{self.q_updates}\n")

        print(f"[EPISODE END #{getattr(self,'episodesSoFar',0)}] Reward={total}, QUpdates={self.q_updates}")

        self.q_updates = 0



############################################################
# PacmanQAgent
############################################################

class PacmanQAgent(QLearningAgent):

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):

        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining

        self.index = 0
        QLearningAgent.__init__(self, **args)

        print(f"[INIT] PacmanQAgent ε={epsilon}, γ={gamma}, α={alpha}, train={numTraining}")

    def getAction(self, state):
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

    def final(self, state):
        QLearningAgent.final(self, state)



############################################################
# ApproximateQAgent
############################################################

class ApproximateQAgent(PacmanQAgent):

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        print(f"[INIT] ApproximateQAgent using extractor: {extractor}")

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        feats = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[f] * v for f, v in feats.items())

    def update(self, state, action, nextState, reward):

        feats = self.featExtractor.getFeatures(state, action)
        current = self.getQValue(state, action)
        nextVal = self.computeValueFromQValues(nextState)

        target = reward + self.discount * nextVal
        diff = target - current

        for f, v in feats.items():
            self.weights[f] += self.alpha * diff * v

    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            print("[TRAINING COMPLETE] Final Weights:", dict(self.weights))


class HybridAStarQAgent(ApproximateQAgent):
    """
    Hybrid A* + Approximate Q-Learning agent.
    Uses SimpleExtractor + A* consistency feature.
    """
    def __init__(self, **args):
        super().__init__(**args)
        self.explorationFunction = None  # if needed
