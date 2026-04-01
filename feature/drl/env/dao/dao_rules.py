from dataclasses import dataclass

@dataclass
class RuleInput:
    action: int
    price: float
    cash: float
    position: float

# Hàm sử dụng: rules.py/apply, env.py/step, reward.py/calculate