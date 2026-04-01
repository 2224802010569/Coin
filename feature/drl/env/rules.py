from typing import Dict, Tuple
from feature.drl.env.config import ACTION_MAP, EnvConfig
from feature.drl.env.entities import TradeResult
from feature.drl.env.dao.dao_rules import RuleInput

class TradingRules:
    def __init__(self, config: EnvConfig):
        self.config = config

    def valid_actions(self, cash: float, pos: float, price: float):
        state = self.account_state(cash, pos, price)
        can_buy = (
            price > 0
            and cash > price * 0.0001
            and state["cash_ratio"] > self.config.min_cash_ratio + self.config.rebalance_tolerance
            and state["asset_ratio"] < self.config.buy_target_exposure - self.config.rebalance_tolerance
        )
        can_sell = (
            pos > 0
            and state["asset_ratio"] > self.config.sell_target_exposure + self.config.rebalance_tolerance
        )
        return {
            0: can_sell,
            1: True,
            2: can_buy,
        }

    def recommended_actions(self, cash: float, pos: float, price: float) -> list[int]:
        state = self.account_state(cash, pos, price)
        if state["cash_ratio"] < self.config.min_cash_ratio:
            return [0, 1]
        if state["asset_ratio"] < self.config.sell_target_exposure:
            return [2, 1]
        if state["asset_ratio"] > self.config.buy_target_exposure:
            return [0, 1]
        return [1, 2, 0]
    
    def apply(self, data: RuleInput) -> Tuple[float, float, TradeResult]:
        action_name = ACTION_MAP.get(int(data.action), "hold")

        if action_name == "buy":
            return self._rebalance_for_action(
                price=data.price,
                cash=data.cash,
                pos=data.position,
                target_exposure=self.config.buy_target_exposure,
                action_name="buy",
            )
        if action_name == "sell":
            return self._rebalance_for_action(
                price=data.price,
                cash=data.cash,
                pos=data.position,
                target_exposure=self.config.sell_target_exposure,
                action_name="sell",
            )
        
        return data.cash, data.position, TradeResult(action_name="hold")

    def _rebalance_for_action(
        self,
        price: float,
        cash: float,
        pos: float,
        target_exposure: float,
        action_name: str,
    ) -> Tuple[float, float, TradeResult]:
        if price <= 0:
            return cash, pos, TradeResult(action_name=action_name, invalid=True)

        state = self.account_state(cash, pos, price)
        total_value = state["portfolio_value"]
        if total_value <= 0:
            return cash, pos, TradeResult(action_name=action_name, invalid=True)

        target_exposure = min(
            max(target_exposure, 0.0),
            max(0.0, 1.0 - self.config.min_cash_ratio),
        )
        desired_asset_value = total_value * target_exposure
        current_asset_value = state["position_value"]
        delta_value = desired_asset_value - current_asset_value

        if abs(delta_value) / max(total_value, 1e-8) <= self.config.rebalance_tolerance:
            return cash, pos, TradeResult(action_name=action_name, invalid=True)

        if delta_value > 0:
            return self._execute_buy(price, cash, pos, min(delta_value, cash), action_name)
        return self._execute_sell(price, cash, pos, min(abs(delta_value), current_asset_value), action_name)

    def _execute_buy(
        self,
        price: float,
        cash: float,
        pos: float,
        amount_to_spend: float,
        action_name: str,
    ) -> Tuple[float, float, TradeResult]:
        max_spend = max(0.0, cash - (cash + pos * price) * self.config.min_cash_ratio)
        amount_to_spend = min(max(amount_to_spend, 0.0), max_spend)
        if amount_to_spend <= 0:
            return cash, pos, TradeResult(action_name=action_name, invalid=True)

        fee = amount_to_spend * self.config.trading_fee
        units = max(0.0, amount_to_spend - fee) / price
        res = TradeResult(action_name=action_name, traded_cash=amount_to_spend, traded_units=units, fee_paid=fee)
        return cash - amount_to_spend, pos + units, res

    def _execute_sell(
        self,
        price: float,
        cash: float,
        pos: float,
        value_to_sell: float,
        action_name: str,
    ) -> Tuple[float, float, TradeResult]:
        units_to_sell = min(max(value_to_sell / price, 0.0), pos)
        if units_to_sell <= 0:
            return cash, pos, TradeResult(action_name=action_name, invalid=True)

        gross_proceeds = units_to_sell * price
        fee = gross_proceeds * self.config.trading_fee
        net_proceeds = gross_proceeds - fee
        res = TradeResult(action_name=action_name, traded_cash=net_proceeds, traded_units=units_to_sell, fee_paid=fee)
        return cash + net_proceeds, pos - units_to_sell, res

    @staticmethod
    def position_value(price: float, position: float) -> float:
        return max(price, 0.0) * max(position, 0.0)

    def portfolio_value(self, price: float, cash: float, position: float) -> float:
        return cash + self.position_value(price, position)

    def trade_summary(self, res: TradeResult) -> Dict[str, float | str | bool]:
        return {
            "action_name": res.action_name,
            "traded_cash": res.traded_cash,
            "traded_units": res.traded_units,
            "fee_paid": res.fee_paid,
            "invalid_action": res.invalid,
        }

    def account_state(self, cash: float, pos: float, price: float) -> Dict[str, float]:
        position_value = self.position_value(price, pos)
        portfolio_value = cash + position_value
        safe_total = max(portfolio_value, 1e-8)
        return {
            "cash": float(cash),
            "position": float(pos),
            "position_value": float(position_value),
            "portfolio_value": float(portfolio_value),
            "cash_ratio": float(cash / safe_total),
            "asset_ratio": float(position_value / safe_total),
        }
