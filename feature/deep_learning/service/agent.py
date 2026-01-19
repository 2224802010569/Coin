import numpy as np

class AgentService:
    def __init__(self, action_size=3, epsilon=0.1):
        self.action_size = action_size
        self.epsilon = epsilon
        self.memory = []

    def act(self, state):
        # random policy (baseline)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.random.randint(self.action_size)

    def learn(self, state, action, reward, next_state, done):
        # lưu kinh nghiệm, sprint sau mới dùng
        self.memory.append((state, action, reward, next_state, done))
