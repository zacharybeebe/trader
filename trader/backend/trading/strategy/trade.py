from ....utils.utils import *

class Trade:
        def __init__(
            self,
            opened: datetime,
            cash: float,
            buy_price: float,
            commission_pct: float = 0
        ):
            self.opened = opened
            self.cash = float(cash)

            self.buy_price = buy_price
            self.shares = self.cash / self.buy_price

            self.min_price = float(self.buy_price)
            self.max_price = float(self.buy_price)
            self.closed = None
            self.sell_price = None
            self.commission_pct = commission_pct
            self.commission = None
            self.pnl = None
            self.pnl_pct = None
            self.duration = None
            self.is_stop_loss = False
            self.is_take_profit = False
            self.is_end_position = False
        
        def close(
            self, 
            closed: datetime, 
            sell_price: float, 
            is_stop_loss: bool = False, 
            is_take_profit: bool = False,
            is_end_position: bool = False
        ) -> None:
            self.closed = closed
            self.sell_price = sell_price
            gross_sell = self.shares * self.sell_price
            self.commission = gross_sell * (self.commission_pct / 100)
            net_sell = gross_sell - self.commission
            self.pnl = net_sell - self.cash
            self.pnl_pct = (self.pnl / self.cash) * 100
            self.cash = net_sell
            self.duration = self.closed - self.opened
            self.is_stop_loss = is_stop_loss
            self.is_take_profit = is_take_profit
            self.is_end_position = is_end_position

        def update_min_max_price(self, price: float):
            if price < self.min_price:
                self.min_price = price
            if price > self.max_price:
                self.max_price = price
        
        def net_value(self, price: float) -> float:
            gross_sell = self.shares * price
            commission = gross_sell * (self.commission_pct / 100)
            return gross_sell - commission
        
        def gross_value(self, price: float) -> float:
            return self.shares * price