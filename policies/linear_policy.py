import torch
import numpy as np
from torch.distributions import Categorical


class LinearPolicy:
    def __init__(self, state_size=2, action_size=4, maze_size=None):
        self.action_size = action_size
        self.maze_size = np.array(maze_size if maze_size is not None else [1,1], dtype=np.float32)
        self.W = torch.zeros(state_size + 1, action_size, requires_grad=True)

    def _phi(self, state):
        s = np.array(state, dtype=np.float32) / self.maze_size
        phi = np.append(s, 1.0)
        return torch.from_numpy(phi).float()

    def act(self, state):
        phi = self._phi(state)
        logits = phi @ self.W
        probs = torch.softmax(logits, dim=0)
        dist  = Categorical(probs)
        a     = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy()
        #return a.item(), dist.log_prob(a)
    def parameters(self):
        return [self.W]