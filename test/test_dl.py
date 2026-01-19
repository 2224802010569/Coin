import sys
import os
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from feature.data.entities.candle import Candle
from feature.label.entities.trend import Trend
from feature.deep_learning.input.label_input import LabelInput
from feature.deep_learning.service.split import SplitService
from feature.deep_learning.input.processing_input import ProcessingInput
from feature.deep_learning.service.processor import ProcessorService
from feature.deep_learning.service.env import EnvService
from feature.deep_learning.service.agent import AgentService
from feature.deep_learning.service.train import TrainService
from feature.deep_learning.service.eval import EvalService
from feature.deep_learning.service.registry import RegistryService
from test_csv import csv_pd


# df = ProcessingInput().run()
# splits = SplitService().run(df=df)
# processor = ProcessorService()
# env = EnvService()
# agent = AgentService()
# for window in splits[:1]:
#     x1, x2 = processor.run(window)
#     state = None
#     while True:
#         if state is None:
#             state = x1[0]
#         action = agent.act(state)
#         next_state, reward, done, _ = env.run(action, x1)
#         agent.learn(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             break
# print(len(agent.memory))

registry = RegistryService()
profile_safe = {
    "name": "safe",
    "eval": "stability"
}
df = ProcessingInput().run()
splits = SplitService().run(df=df)
train = TrainService()
train.run(splits[:2])
metrics = EvalService().run(train.rewards)
registry.save(train.agent, metrics, profile_safe)
best = registry.select(profile_safe)
print(best["metrics"])