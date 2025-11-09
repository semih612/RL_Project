import numpy as np
import random


class MultiAgentMazeEnv:
    def __init__(self, size=(5, 5), starts=[(0, 0), (0, 4)], goals=[(5, 5), (5, 5)], walls=None):
        self.size = size
        self.starts = starts
        self.goals = goals
        self.walls = walls if walls else []
        self.n_agents = len(starts)
        self.reset()

    def reset(self):
        self.agent_positions = [list(start) for start in self.starts]
        # Track last positions to detect "staying still"
        self.last_positions = [list(start) for start in self.starts]
        return [tuple(pos) for pos in self.agent_positions]

    def step(self, actions, weights=None):
        """
        actions: list of integers (0=up, 1=down, 2=left, 3=right)
        returns: new positions, rewards, dones
        """
        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        n = len(actions)
        order = list(range(n))  # default order
        
        if weights is None:
            weights = np.ones(n, dtype=float)  # equal weights
        else:
            weights = np.array(weights, dtype=float)
            
    
        if weights is not None:
            # --- Weighted random order (no replacement) ---
            order = []
            remaining_agents = list(range(n))
            remaining_weights = np.array(weights, dtype=float).copy()
    
            while remaining_agents:
                probs = remaining_weights / remaining_weights.sum()
                idx = np.random.choice(len(remaining_agents), p=probs)
                order.append(remaining_agents.pop(idx))
                remaining_weights = np.delete(remaining_weights, idx)

        new_positions = []
        next_pos = []
        for i in range(0,n): 
            new_positions.append(tuple(map(int,[-1,-1])))
            next_pos.append(tuple(map(int,[-1,-1])))

        agent_pos_temp = np.array(self.agent_positions, dtype=float).copy()
        for i in order:   # process agents in weighted-random order
            action = actions[i]
            move = moves[action]
            next_pos[i] = tuple(map(int,[agent_pos_temp[i][0] + move[0],
                       agent_pos_temp[i][1] + move[1]]))

            # check bounds and walls
            if (0 <= next_pos[i][0] < self.size[0] and 0 <= next_pos[i][1] < self.size[1] and next_pos[i] not in self.walls):
                if next_pos[i] not in new_positions:
                    new_positions[i] = next_pos[i]
                elif next_pos[i] in new_positions:
                    new_positions[i] = tuple(map(int,self.agent_positions[i]))
            else:
                if tuple(agent_pos_temp[i]) in new_positions:
                    new_positions = list(self.agent_positions)
                    break
                else:
                    new_positions[i] = tuple(map(int,self.agent_positions[i]))
        
        self.last_positions = [tuple(p) for p in self.agent_positions]  # remember old positions
        self.agent_positions = new_positions
        
        rewards = np.zeros(n, dtype=float)
        dones   = np.zeros(n, dtype=bool)  
        # who reached their goal this step?
        reached = [i for i in range(n) if tuple(new_positions[i]) == self.goals[i]]

        
        # --- New Blocking Logic ---
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Agent i stayed still
                    if tuple(self.agent_positions[i]) == tuple(self.last_positions[i]):
                        # If opponent moved into that position -> blocking success
                        if next_pos[j] == tuple(self.agent_positions[i]):
                            rewards[i] -= 0.5   # small blocking reward
                            rewards[j] += 0.5   # symmetric penalty

                            dones[:] = True 
                            return new_positions, rewards, dones
                            

        
        if len(reached) == 1:
            w = reached[0]
            rewards[w] = 1
            for j in range(n):
                if j != w:
                    rewards[j] = -1
            dones[:] = True  # end episode after a win
        elif len(reached) >= 2:
            # simultaneous arrival: tie → zeros, end episode
            rewards[:] = 0.0
            dones[:] = True
        else:
            # no one reached: keep going
            dones[:] = False
        
        return new_positions, rewards, dones

    def render(self):
        grid = np.full(self.size, " ", dtype=object)
    
        # Draw walls
        for w in self.walls:
            grid[w] = "#"
    
        # Draw numbered goals (colored numbers)
        colors = ["\033[92m", "\033[94m", "\033[91m", "\033[93m"]  # green, blue, red, yellow
        for i, g in enumerate(self.goals):
            color = colors[i % len(colors)]
            grid[g] = f"{color}{i}\033[0m"
    
        # Draw numbered agents
        for i, pos in enumerate(self.agent_positions):
            grid[tuple(pos)] = f"{i}"
    
        # Print
        for row in grid:
            print(" ".join(row))
        print("----------")


def random_maze_env(size=(5,5), n_agents=1, n_walls=6, seed=None):
    """
    Build a random MultiAgentMazeEnv with given size, agents, and wall count.
    
    Args:
        size (tuple): grid size (rows, cols)
        n_agents (int): number of agents
        n_walls (int): number of wall cells
        seed (int or None): random seed for reproducibility
    
    Returns:
        env: a MultiAgentMazeEnv instance
    """
    if seed is not None:
        random.seed(seed)
    else:
        seed = random.randint(0, int(1e9))
        random.seed(seed)
        print("Generated seed:", seed)

    rows, cols = size

    # --- choose starts and goals ---
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    starts = random.sample(all_cells, n_agents)
    remaining = [c for c in all_cells if c not in starts]
    goals = random.sample(remaining, n_agents)

    # --- choose walls (avoid starts + goals) ---
    forbidden = set(starts + goals)
    candidates = [c for c in all_cells if c not in forbidden]
    walls = random.sample(candidates, min(n_walls, len(candidates)))

    # --- build env ---
    env = MultiAgentMazeEnv(
        size=size,
        starts=starts,
        goals=goals,
        walls=walls
    )
    return env

