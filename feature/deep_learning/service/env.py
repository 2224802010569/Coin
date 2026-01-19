import numpy as np
from config import COIN

class EnvService:
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2

    def __init__(self):
        self.prices = None
        self.cost_rate = COIN.FEE
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 0
        self.entry_price = 0.0
        self.done = False

    def _state(self):
        return self.prices[self.t]

    def run(self, action, x_train):
        # init prices ONCE
        if self.prices is None:
            self.prices = x_train
            self.reset()

        if self.done:
            raise RuntimeError("Episode done")

        reward = 0.0
        price = self.prices[self.t][0]

        if action != self.position:
            reward -= self.cost_rate

        # pnl
        if self.position != 0:
            reward += self.position * (price - self.entry_price)

        # update position
        if action != self.position:
            self.entry_price = price
            self.position = action if action != self.ACTION_HOLD else 0

        self.t += 1
        if self.t >= len(self.prices) - 1:
            self.done = True

        return self._state(), reward, self.done, {}
