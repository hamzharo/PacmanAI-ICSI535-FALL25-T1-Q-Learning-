# # featureExtractors.py
# # --------------------
# # Licensing Information:  You are free to use or extend these projects for
# # educational purposes provided that (1) you do not distribute or publish
# # solutions, (2) you retain this notice, and (3) you provide clear
# # attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# #
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# # The core projects and autograders were primarily created by John DeNero
# # (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # Student side autograding was added by Brad Miller, Nick Hay, and
# # Pieter Abbeel (pabbeel@cs.berkeley.edu).


# "Feature extractors for Pacman game states"

# from game import Directions, Actions
# import util

# class FeatureExtractor:
#     def getFeatures(self, state, action):
#         """
#           Returns a dict from features to counts
#           Usually, the count will just be 1.0 for
#           indicator functions.
#         """
#         util.raiseNotDefined()

# class IdentityExtractor(FeatureExtractor):
#     def getFeatures(self, state, action):
#         feats = util.Counter()
#         feats[(state,action)] = 1.0
#         return feats

# class CoordinateExtractor(FeatureExtractor):
#     def getFeatures(self, state, action):
#         feats = util.Counter()
#         feats[state] = 1.0
#         feats['x=%d' % state[0]] = 1.0
#         feats['y=%d' % state[0]] = 1.0
#         feats['action=%s' % action] = 1.0
#         return feats

# def closestFood(pos, food, walls):
#     """
#     closestFood -- this is similar to the function that we have
#     worked on in the search project; here its all in one place
#     """
#     fringe = [(pos[0], pos[1], 0)]
#     expanded = set()
#     while fringe:
#         pos_x, pos_y, dist = fringe.pop(0)
#         if (pos_x, pos_y) in expanded:
#             continue
#         expanded.add((pos_x, pos_y))
#         # if we find a food at this location then exit
#         if food[pos_x][pos_y]:
#             return dist
#         # otherwise spread out from the location to its neighbours
#         nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
#         for nbr_x, nbr_y in nbrs:
#             fringe.append((nbr_x, nbr_y, dist+1))
#     # no food found
#     return None

# class SimpleExtractor(FeatureExtractor):
#     """
#     Returns simple features for a basic reflex Pacman:
#     - whether food will be eaten
#     - how far away the next food is
#     - whether a ghost collision is imminent
#     - whether a ghost is one step away
#     """

#     def getFeatures(self, state, action):
#         # extract the grid of food and wall locations and get the ghost locations
#         food = state.getFood()
#         walls = state.getWalls()
#         ghosts = state.getGhostPositions()

#         features = util.Counter()

#         features["bias"] = 1.0

#         # compute the location of pacman after he takes the action
#         x, y = state.getPacmanPosition()
#         dx, dy = Actions.directionToVector(action)
#         next_x, next_y = int(x + dx), int(y + dy)

#         # count the number of ghosts 1-step away
#         features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

#         # if there is no danger of ghosts then add the food feature
#         if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
#             features["eats-food"] = 1.0

#         dist = closestFood((next_x, next_y), food, walls)
#         if dist is not None:
#             # make the distance a number less than one otherwise the update
#             # will diverge wildly
#             features["closest-food"] = float(dist) / (walls.width * walls.height)
#         features.divideAll(10.0)
#         return features




# featureExtractors.py
# --------------------
# Original Berkeley feature extractors, extended with an
# A*-consistency feature for SimpleExtractor so that
# Approximate Q-Learning can be guided by A*.

import util
from game import Directions, Actions
from astar_helper import astar_path_to_closest_food


########################################
# Base class
########################################

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        Given a state and action, return a Counter of feature_name -> value.
        """
        util.raiseNotDefined()


########################################
# Identity Extractor (unchanged)
########################################

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


########################################
# Simple Extractor + A* Consistency
########################################

def closestFood(pos, food, walls):
    """
    Helper used by the original SimpleExtractor to compute the distance
    to the nearest food using BFS.
    """
    from util import Queue
    fringe = Queue()
    expanded = set()
    fringe.push((pos, 0))
    while not fringe.isEmpty():
        (x, y), dist = fringe.pop()
        if (x, y) in expanded:
            continue
        expanded.add((x, y))
        if food[x][y]:
            return dist
        # explore neighbors
        for nx, ny in Actions.getLegalNeighbors((x, y), walls):
            if (nx, ny) not in expanded:
                fringe.push(((nx, ny), dist + 1))
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Original CS188 SimpleExtractor, extended with:
      - feature "aStar-consistency": 1.0 if the action follows the
        next step on the A* path to the closest food; 0.0 otherwise.

    This lets Approximate Q-Learning learn when to follow the A* plan
    and when to deviate because of ghosts or other risks.
    """

    def __init__(self):
        # Cache last state's A* path so we don't recompute for every action
        self._astar_cache_state = None
        self._astar_cache_path = []

    def _ensure_astar_cache(self, state):
        """
        Recompute A* path to closest food if state changed.
        """
        if self._astar_cache_state is state:
            return

        self._astar_cache_state = state
        self._astar_cache_path = astar_path_to_closest_food(state)

    def getFeatures(self, state, action):
        # Start from the original features
        features = util.Counter()

        # Extract the grids
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        # Compute the position of Pacman after taking the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # 1) Ghost proximity feature (original)
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts
        )

        # 2) Closest food feature (original)
        if not food[next_x][next_y] and features["#-of-ghosts-1-step-away"] == 0:
            dist = closestFood((next_x, next_y), food, walls)
            if dist is not None:
                # Normalize by map area
                features["closest-food"] = float(dist) / (walls.width * walls.height)

        # 3) NEW: A* consistency feature
        self._ensure_astar_cache(state)
        astar_path = self._astar_cache_path  # list of (x,y) positions

        a_star_next = astar_path[0] if len(astar_path) > 0 else None
        if a_star_next is not None and (next_x, next_y) == a_star_next:
            features["aStar-consistency"] = 1.0
        else:
            features["aStar-consistency"] = 0.0

        # Scale all features down a bit (same as original)
        features.divideAll(10.0)
        return features


########################################
# (Optional) CoordinateExtractor
# If your project uses this, keep or fix it here.
########################################

class CoordinateExtractor(FeatureExtractor):
    """
    Example coordinate-based feature extractor.
    If you don't use this in your project, it's harmless to leave here.
    """

    def getFeatures(self, state, action):
        feats = util.Counter()
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        feats["x=%d" % next_x] = 1.0
        feats["y=%d" % next_y] = 1.0
        return feats
