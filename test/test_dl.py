from profile import Profile
import sqlite3
import sys
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from feature.drl.usecase.backtest_uc import BacktestUsecase
from feature.drl.usecase.train_uc import TrainAgentUsecase

trainer = TrainAgentUsecase()
trainer.run("safe")

# backtester = BacktestUsecase()
# backtester.run("safe")