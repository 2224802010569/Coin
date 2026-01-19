from feature.deep_learning.service.agent import AgentService
from feature.deep_learning.service.env import EnvService
from feature.deep_learning.service.processor import ProcessorService


class TrainService:
    def __init__(self):
        self.agent = AgentService()
        self.processor = ProcessorService()
        self.rewards = []

    def run(self, splits):
        for window in splits:
            env = EnvService()
            x_train, _ = self.processor.run(window)
            state = None
            while True:
                if state is None:
                    state = x_train[0]
                action = self.agent.act(state)
                next_state, reward, done, _ = env.run(action, x_train)
                self.agent.learn(state, action, reward, next_state, done)
                self.rewards.append(reward)
                state = next_state
                if done:
                    break
