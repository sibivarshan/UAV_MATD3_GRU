import numpy as np

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.copy(self.mu)
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

class GaussianNoise:
    def __init__(self, size, std_start=0.2, std_end=0.05, decay_steps=10000):
        self.size = size
        self.std_start = std_start
        self.std_end = std_end
        self.decay_steps = decay_steps
        self.current_std = std_start
        self.steps = 0
    
    def sample(self):
        self.steps += 1
        # Linear decay
        self.current_std = max(
            self.std_end,
            self.std_start - (self.std_start - self.std_end) * (self.steps / self.decay_steps)
        )
        return np.random.randn(self.size) * self.current_std
    
    def reset(self):
        pass