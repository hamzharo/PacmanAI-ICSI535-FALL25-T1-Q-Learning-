# astar_helper.py
# ---------------
# A* helper functions used as guidance for Approximate Q-Learning.
#
# We use this to compute a shortest path from Pacman's current
# position to the nearest food dot, ignoring ghosts. The resulting
# path is then used as a feature inside SimpleExtractor.

from util import PriorityQueue, manhattanDistance


def astar_path_to_closest_food(gameState):
    """
    Run A* search on the grid to find a path from Pacman's current
    position to the closest food dot.

    Returns:
        path: list of (x, y) positions, NOT including the start.
              If no food or no path, returns [].
    """
    start = gameState.getPacmanPosition()
    walls = gameState.getWalls()
    foodGrid = gameState.getFood()
    food_positions = foodGrid.asList()

    # No food = no path
    if not food_positions:
        return []

    # Heuristic: Manhattan distance to the closest food
    def heuristic(pos):
        return min(manhattanDistance(pos, f) for f in food_positions)

    # Neighbor generator
    def get_neighbors(position):
        x, y = position
        neighbors = []
        # 4-connected grid
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                neighbors.append((nx, ny))
        return neighbors

    # A* over grid positions
    frontier = PriorityQueue()
    # Each item in frontier: (position, path_positions)
    # path_positions is a list of positions AFTER start
    frontier.push((start, []), heuristic(start))
    visited = set()

    while not frontier.isEmpty():
        position, path = frontier.pop()

        if position in visited:
            continue
        visited.add(position)

        # Goal test: reached some food (not counting start)
        if position in food_positions and position != start:
            return path

        for next_pos in get_neighbors(position):
            if next_pos in visited:
                continue
            new_path = path + [next_pos]
            cost = len(new_path)
            priority = cost + heuristic(next_pos)
            frontier.push((next_pos, new_path), priority)

    # No path found
    return []
