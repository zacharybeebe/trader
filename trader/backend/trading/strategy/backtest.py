from .broker import *


class BackTest(Broker):    
    def __init__(
        self, 
        strategy: 'Strategy',
        cash: float,
        commission: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None, 
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False,
        close_end_position: bool = True,     
    ):
        self.strategy = strategy
        self.data = self.strategy.data
        self._ran = False

        super().__init__(
            start_price=self.data['close'].iloc[0],
            start_date=self.data.index[0],
            cash=cash,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit,
            restrict_sell_below_buy=restrict_sell_below_buy,
            restrict_non_profitable=restrict_non_profitable,
            close_end_position=close_end_position
        )    
        
    def __repr__(self):
        text = '#' * 100 + '\n'
        text += f'{self.strategy.__class__.__name__} BackTest\n'
        text += self._prt_outcomes()
        return text

    def run(self):
        buy_col = self.strategy.buy_column
        sell_col = self.strategy.sell_column

        current_cash = float(self.initial_cash)
        current_trade = None

        # Only get the data that have either buy or sell signals
        # and shrink the size of the dataframe to only the necessary columns
        signal_data = self.data.loc[(
            (self.data[buy_col].notna()) 
            | (self.data[sell_col].notna())
        ), ['close', buy_col, sell_col]]

        for row in signal_data.itertuples():
            dt = row.Index
            current_price = row.close
            buy_signal = getattr(row, buy_col)
            sell_signal = getattr(row, sell_col)

            current_trade, current_cash = self.evaluate(
                at_price=current_price,
                at_date=dt,
                buy_signal=buy_signal,
                sell_signal=sell_signal,
                current_cash=current_cash,
                current_trade=current_trade
            )

        self.complete(
            end_price=self.data['close'].iloc[-1],
            end_date=self.data.index[-1],
            current_cash=current_cash,
            current_trade=current_trade
        )
        self._ran = True