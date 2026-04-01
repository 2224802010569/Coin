import numpy as np
import pandas as pd
from feature.drl.env.dao.dao_rules import RuleInput
from feature.drl.env.dao.dao_reward import RewardInput

def validate_and_clean_df(df: pd.DataFrame, columns: list, price_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    required = set(columns) | {price_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
        
    df = df.copy()
    df[list(required)] = df[list(required)].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return df

def get_window_data(df: pd.DataFrame, step: int, window_size: int, cols: list) -> np.ndarray:
    start = max(0, step - window_size + 1)
    window = df.loc[start : step, list(cols)].to_numpy(dtype=np.float32)    
    if len(window) < window_size:
        pad_width = ((window_size - len(window), 0), (0, 0))
        window = np.pad(window, pad_width, mode="edge")
    return window.reshape(-1)

def portfolio_breakdown(price, cash, pos):
    position_value = max(price, 0.0) * max(pos, 0.0)
    total_value = cash + position_value
    safe_total = max(total_value, 1e-8)
    return {
        "position_value": position_value,
        "total_value": total_value,
        "cash_ratio": cash / safe_total,
        "asset_ratio": position_value / safe_total,
    }

def normalize_state_features(cash, pos, price, value, peak_value, step, total_steps, init_bal) -> np.ndarray:
    safe_bal = max(init_bal, 1e-8)
    breakdown = portfolio_breakdown(price, cash, pos)
    drawdown = max(0.0, (peak_value - value) / max(peak_value, 1e-8))
    return np.array([
        breakdown["cash_ratio"],
        breakdown["asset_ratio"],
        breakdown["position_value"] / safe_bal,
        value / safe_bal,
        drawdown,
        step / max(total_steps, 1)
    ], dtype=np.float32)

def ensure_not_done(env):
    if env.done:
        raise RuntimeError("TradingEnv.step() called after episode is done.")

def prepare_step_data(env):
    price = env._current_price()
    prev_val = env.rules.portfolio_value(price, env.cash, env.position)
    return price, prev_val

def apply_rules_wrapper(env, action, price):
    inp = RuleInput(action=action, price=price, cash=env.cash, position=env.position)
    env.cash, env.position, trade_res = env.rules.apply(inp)
    return env.cash, env.position, trade_res

def update_state_after_trade(env):
    env.current_step = min(env.current_step + 1, len(env.df) - 1)
    env.done = env.current_step >= len(env.df) - 1
    curr_val = env.rules.portfolio_value(env._current_price(), env.cash, env.position)
    env.peak_value = max(env.peak_value, curr_val)
    return curr_val

def compute_reward_wrapper(env, prev_val, curr_val, trade_res):
    inp = RewardInput(
        prev_val=prev_val,
        curr_val=curr_val,
        peak_value=env.peak_value,
        fee_paid=trade_res.fee_paid,
        invalid=trade_res.invalid,
        action_name=trade_res.action_name,
    )
    return float(env.reward_calculator.calculate(inp))
