from dataclasses import dataclass

@dataclass
class RewardInput:
    prev_val: float
    curr_val: float
    peak_value: float
    fee_paid: float
    invalid: bool
    action_name: str

# Hàm sử dụng: reward.py/calculate, rules.py/apply, env.py/step