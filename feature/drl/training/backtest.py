from __future__ import annotations

import pandas as pd

from feature.drl.agent.agent import DQNAgent
from feature.drl.env.env import TradingEnv
from feature.drl.observation.processor import ObservationProcessor
from feature.drl.training.config import TrainingConfig
from feature.drl.training.logger import TrainingLogger
from feature.drl.training import utils


class Backtest:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.logger = TrainingLogger(width=self.config.progress_bar_width)
        self.processor = ObservationProcessor()

    def run(self) -> pd.DataFrame:
        df = self._load_data()
        env = TradingEnv(df)
        agent = self._load_agent(env, df)
        records = self._run_steps(env, agent, df)
        result = pd.DataFrame(records)[self.config.cols]
        # result = pd.DataFrame(records)
        self.logger.info(f"Backtest complete: {len(result)} rows")
        return result

    def _load_data(self):
        from feature.processing.usecase.processing_uc import ProcessingUC
        return ProcessingUC().run(tf=self.config.train_timeframe).copy()

    def _load_agent(self, env, df):
        state = self.processor.transform(env.reset())
        agent = DQNAgent(state_size=len(state), action_size=env.action_space)
        agent.load(self.config.model_path, strict=False)
        return agent

    def _run_steps(self, env, agent, df):
        records = []
        max_steps = max(1, len(df) - env.current_step)
        state = self.processor.transform(env.reset())

        for step in range(max_steps):
            record, state = self._run_single_step(env, agent, df, state)

            records.append(record)
            suffix = (
                f"portfolio={record['portfolio_value']:.4f} "
                f"cash={record['cash']:.4f} "
                f"position={record['position']:.6f}"
            )
            utils.update_progress(self.logger, "Backtest", step + 1, max_steps, suffix)

            if record["done"]:
                break

        return records

    def _run_single_step(self, env, agent, df, state):
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        state = self.processor.transform(next_state)
        row_index = env.current_step
        record = df.iloc[row_index].to_dict()
        record.update(info)
        record["reward"] = reward
        return record, state
