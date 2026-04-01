# env.py (refactor)

from __future__ import annotations
import numpy as np
import pandas as pd

from feature.drl.env.config import ACTION_MAP, EnvConfig
from feature.drl.env.entities import TradeResult
from feature.drl.env.reward import RewardCalculator
from feature.drl.env.rules import TradingRules
from feature.drl.env.dao.dao_rules import RuleInput
from feature.drl.env.dao.dao_reward import RewardInput
from feature.drl.env import utils

class TradingEnv:
    def __init__(self, df, config: EnvConfig | None = None):
        self.config = config or self._default_config()
        self.df = utils.validate_and_clean_df(df, self.config.observation_columns, self.config.price_column)

        self.rules = TradingRules(self.config)
        self.reward_calculator = RewardCalculator(self.config)

        self.action_space = len(ACTION_MAP)
        self.observation_size = (
            len(self.config.observation_columns) * self.config.window_size + 6 + 3
        )
        self._reset_internal_state()

    def _default_config(self):
        from config import COIN, MODEL
        return EnvConfig(
            window_size=getattr(MODEL, "WINDOW", 16),
            initial_balance=getattr(COIN, "BALANCE", 100.0),
            trading_fee=getattr(COIN, "FEE", 0.001),
        )

    def _reset_internal_state(self):
        self.current_step = 0
        self.cash = float(self.config.initial_balance)
        self.position = 0.0
        self.peak_value = 0.0
        self.done = False

    def _current_price(self) -> float:
        return float(self.df.iloc[self.current_step][self.config.price_column])

    def reset(self) -> np.ndarray:
        self._reset_internal_state()
        self.current_step = min(self.config.window_size - 1, len(self.df) - 1)
        self.peak_value = self.rules.portfolio_value(
            self._current_price(), self.cash, self.position
        )
        return self._get_observation()

    def step(self, action: int):
        utils.ensure_not_done(self)

        price = self._current_price()
        valid_actions = self.rules.valid_actions(self.cash, self.position, price)

        if not valid_actions.get(action, False):
            trade_res = TradeResult("invalid")
            next_val = self._after_skip(price)
            reward = 0.0
        else:
            price, prev_val = utils.prepare_step_data(self)
            cash, pos, trade_res = utils.apply_rules_wrapper(self, action, price)
            next_val = utils.update_state_after_trade(self)
            reward = utils.compute_reward_wrapper(self, prev_val, next_val, trade_res)

        info = self._build_info(action, price, next_val, reward, trade_res, valid_actions)
        return self._get_observation(valid_actions), reward, self.done, info

    # --- NEW: bỏ qua action invalid
    def _after_skip(self, price: float):
        self.current_step = min(self.current_step + 1, len(self.df) - 1)
        self.done = self.current_step >= len(self.df) - 1
        value = self.rules.portfolio_value(price, self.cash, self.position)
        self.peak_value = max(self.peak_value, value)
        return value

    def _get_observation(self, valid=None):
        obs_window = utils.get_window_data(
            self.df, self.current_step, self.config.window_size, self.config.observation_columns
        )
        price = self._current_price()
        value = self.rules.portfolio_value(price, self.cash, self.position)

        state_feats = utils.normalize_state_features(
            self.cash,
            self.position,
            price,
            value,
            self.peak_value,
            self.current_step,
            len(self.df),
            self.config.initial_balance,
        )

        if valid is None:
            valid = self.rules.valid_actions(self.cash, self.position, price)

        mask = np.array([int(valid[0]), int(valid[1]), int(valid[2])], dtype=np.float32)
        return np.concatenate((obs_window, state_feats, mask))

    def _build_info(self, action, price, value, reward, trade_res, valid):
        account_state = self.rules.account_state(self.cash, self.position, price)
        info = {
            "step": self.current_step,
            "action": action,
            "action_name": ACTION_MAP.get(action, "hold"),
            "price": price,
            "cash": self.cash,
            "position": self.position,
            "portfolio_value": value,
            "peak_value": self.peak_value,
            "reward": reward,
            "done": self.done,
            "valid_actions": [a for a, ok in valid.items() if ok],
            "recommended_actions": self.rules.recommended_actions(self.cash, self.position, price),
            "cash_ratio": account_state["cash_ratio"],
            "asset_ratio": account_state["asset_ratio"],
        }
        return info
