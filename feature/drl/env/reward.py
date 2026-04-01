from feature.drl.env.config import EnvConfig
from feature.drl.env.dao.dao_reward import RewardInput

class RewardCalculator:
    def __init__(self, config: EnvConfig):
        self.config = config

    def calculate(self, data: RewardInput) -> float:
        """Hàm chính điều phối việc tính toán tổng reward."""
        reward = self._calculate_base_return(data.prev_val, data.curr_val)
        reward -= self._calculate_drawdown_penalty(data.peak_value, data.curr_val)
        reward -= self._calculate_transaction_cost(data.fee_paid, data.prev_val)
        reward -= self._calculate_behavioral_penalty(data.invalid, data.action_name)
        return float(reward * self.config.reward_scale)

    def _calculate_base_return(self, prev: float, curr: float) -> float:
        """Tính toán tỷ lệ lợi nhuận (Return)."""
        if prev <= 0:
            return 0.0
        return (curr - prev) / prev

    def _calculate_drawdown_penalty(self, peak: float, curr: float) -> float:
        """Tính toán hình phạt dựa trên mức sụt giảm tài sản từ đỉnh."""
        if peak <= 0:
            return 0.0
        drawdown = max(0.0, (peak - curr) / peak)
        return drawdown * self.config.max_drawdown_penalty

    def _calculate_transaction_cost(self, fee: float, prev: float) -> float:
        """Tính toán tác động của phí giao dịch lên tài sản."""
        return fee / max(prev, 1e-8)

    def _calculate_behavioral_penalty(self, is_invalid: bool, action: str) -> float:
        """Tính toán hình phạt cho các hành động không hợp lệ hoặc 'hold'."""
        if is_invalid:
            return float(self.config.invalid_action_penalty)
        # if action == "hold":
        #     return float(self.config.hold_penalty)
        return 0.0