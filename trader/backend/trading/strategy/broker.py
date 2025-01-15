from .trade import *


class Broker(object):
    _CASH_FMT_FUNC = lambda x: 'N/A' if isna(x) else f'${x:,.4f}' if -1 < x < 1 else f'${x:,.2f}'
    _NUM_FMT_FUNC = lambda x: 'N/A' if isna(x) else f'{x:,.0f}' if isinstance(x, int) else f'{x:,.2f}'
    _PERCENT_FMT_FUNC = lambda x: 'N/A' if isna(x) else f'{x:.4f}%' if -1 < x < 1 else f'{x:.2f}%'
    _DATE_FMT_FUNC = lambda x: 'N/A' if isna(x) else pretty_time(seconds=x.total_seconds()) if isinstance(x, timedelta) else x.strftime('%m-%d-%Y %H:%M')
    _PARAM_ATTRS = {
        'initial_cash': {'func': _CASH_FMT_FUNC, 'label': 'Initial Cash'},
        'commission': {'func': _PERCENT_FMT_FUNC, 'label': 'Commission %'},
        'stop_loss': {'func': _PERCENT_FMT_FUNC, 'label': 'Stop Loss %'},
        'take_profit': {'func': _PERCENT_FMT_FUNC, 'label': 'Take Profit %'},
        'start_date': {'func': _DATE_FMT_FUNC, 'label': 'Start Date'},
        'end_date': {'func': _DATE_FMT_FUNC, 'label': 'End Date'},
        'start_price': {'func': _CASH_FMT_FUNC, 'label': 'Start Price'},
        'end_price': {'func': _CASH_FMT_FUNC, 'label': 'End Price'},
        'duration': {'func': _DATE_FMT_FUNC, 'label': 'Duration'}
    }
    _TRADE_ATTRS = {
        'num_trades': {'func': _NUM_FMT_FUNC, 'label': '# of Trades'},
        'num_buys': {'func': _NUM_FMT_FUNC, 'label': '# of Buys'},
        'num_sells': {'func': _NUM_FMT_FUNC, 'label': '# of Sells'},
        'sells_per_day': {'func': _NUM_FMT_FUNC, 'label': 'Trades per Day'},
        'max_cash': {'func': _CASH_FMT_FUNC, 'label': 'Max Cash'},
        'min_cash': {'func': _CASH_FMT_FUNC, 'label': 'Min Cash'},
        'max_buy_price': {'func': _CASH_FMT_FUNC, 'label': 'Max Buy Price'},
        'min_buy_price': {'func': _CASH_FMT_FUNC, 'label': 'Min Buy Price'},
        'max_sell_price': {'func': _CASH_FMT_FUNC, 'label': 'Max Sell Price'},
        'min_sell_price': {'func': _CASH_FMT_FUNC, 'label': 'Min Sell Price'},
        'exposure': {'func': _DATE_FMT_FUNC, 'label': 'Exposure'},
        'exposure_pct': {'func': _PERCENT_FMT_FUNC, 'label': 'Exposure %'},
        'max_exposure': {'func': _DATE_FMT_FUNC, 'label': 'Max Exposure'},
        'min_exposure': {'func': _DATE_FMT_FUNC, 'label': 'Min Exposure'},
        'num_wins': {'func': _NUM_FMT_FUNC, 'label': '# of Wins'},
        'num_losses': {'func': _NUM_FMT_FUNC, 'label': '# of Losses'},
        'num_pushes': {'func': _NUM_FMT_FUNC, 'label': '# of Pushes'},
        'win_pct': {'func': _PERCENT_FMT_FUNC, 'label': 'Win %'},
        'max_win': {'func': _CASH_FMT_FUNC, 'label': 'Max Win'},
        'max_win_pct': {'func': _PERCENT_FMT_FUNC, 'label': 'Max Win %'},
        'avg_win': {'func': _CASH_FMT_FUNC, 'label': 'Avg Win'},
        'avg_win_pct': {'func': _PERCENT_FMT_FUNC, 'label': 'Avg Win %'},
        'max_loss': {'func': _CASH_FMT_FUNC, 'label': 'Max Loss'},
        'max_loss_pct': {'func': _PERCENT_FMT_FUNC, 'label': 'Max Loss %'},
        'avg_loss': {'func': _CASH_FMT_FUNC, 'label': 'Avg Loss'},
        'avg_loss_pct': {'func': _PERCENT_FMT_FUNC, 'label': 'Avg Loss %'},
        'avg_win_weighted': {'func': _CASH_FMT_FUNC, 'label': 'Avg Win Weighted'},
        'avg_loss_weighted': {'func': _CASH_FMT_FUNC, 'label': 'Avg Loss Weighted'},
        'expected_value': {'func': _CASH_FMT_FUNC, 'label': 'Expected Value'}
    }
    _RESULT_ATTRS = {
        'final_cash': {'func': _CASH_FMT_FUNC, 'label': 'Final Cash'},
        'final_shares': {'func': _NUM_FMT_FUNC, 'label': 'Final Shares'},
        'final_value': {'func': _CASH_FMT_FUNC, 'label': 'Final Value'},
        'end_price': {'func': _CASH_FMT_FUNC, 'label': 'Final Price'},
        'commission_paid': {'func': _CASH_FMT_FUNC, 'label': 'Commission Paid'},
        'return_pct': {'func': _PERCENT_FMT_FUNC, 'label': 'Return %'},
        'apy': {'func': _PERCENT_FMT_FUNC, 'label': 'APY %'},
        'apy_value': {'func': _CASH_FMT_FUNC, 'label': 'APY Value'},
        'buy_hold_return': {'func': _PERCENT_FMT_FUNC, 'label': 'Buy Hold %'},
        'buy_hold_value': {'func': _CASH_FMT_FUNC, 'label': 'Buy Hold Value'}
    }
    
    def __init__(
        self,
        start_price: float,
        start_date: datetime,
        cash: float,
        commission: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None, 
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False,
        close_end_position: bool = True,     
    ):
        self.start_price = start_price
        self.start_date = start_date

        if cash <= 0:
            raise ValueError('Cash must be greater than 0')
        self.initial_cash = cash
        self.commission = commission
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.restrict_sell_below_buy = restrict_sell_below_buy
        self.restrict_non_profitable = restrict_non_profitable
        self.close_end_position = close_end_position

        self._trades_data = {
            'opened': [],
            'closed': [],
            'cash': [],
            'shares': [],
            'buy_price': [],
            'sell_price': [],
            'pnl': [],
            'commission': [],
            'duration': [],
            'max_price': [],
            'min_price': [],
            'is_stop_loss': [],
            'is_take_profit': [],
            'is_end_position': []
        }
        self.trades_data: pd.DataFrame = None

        self.num_trades = 0
        self.num_buys = 0
        self.num_sells = 0
        self.sells_per_day = 0
        self.max_cash = cash
        self.min_cash = cash
        self.max_buy_price = None
        self.min_buy_price = None
        self.max_sell_price = None
        self.min_sell_price = None
        self.max_win = 0
        self.max_win_pct = 0
        self.avg_win = 0
        self.avg_win_pct = 0
        self.max_loss = 0
        self.max_loss_pct = 0
        self.avg_loss = 0
        self.avg_loss_pct = 0
        self.avg_win_weighted = 0
        self.avg_loss_weighted = 0
        self.expected_value = 0
        self.exposure = 0
        self.exposure_pct = np.nan
        self.max_exposure = None
        self.min_exposure = None
        self.final_cash = cash
        self.final_shares = 0
        self.final_value = cash
        self.num_wins = 0
        self.num_losses = 0
        self.num_pushes = 0
        self.win_pct = np.nan
        self.return_pct = np.nan
        self.apy = np.nan
        self.apy_value = cash
        self.commission_paid = 0

        self.trades = []

        self.end_price = None
        self.end_date = None
        self.duration = None
        self.duration_seconds = 0
        self.buy_hold_return = 0
        self.buy_hold_value = 0
        self.buy_hold_return *= 100
    
    def _prt_outcomes(self) -> str:
        text = ''
        for label, dictionary in [['PARAMETERS', self._PARAM_ATTRS], ['TRADE INFO', self._TRADE_ATTRS], ['RESULTS', self._RESULT_ATTRS]]:
            text += f'{label}\n'
            for at_name, at_item in dictionary.items():
                attr = getattr(self, at_name)
                text += f'-- {at_item["label"]:25} {at_item["func"](attr)}\n'
            text += '\n'
        text += '#' * 100 + '\n'
        return text
    
    def __repr__(self) -> str:
        return self._prt_outcomes()
    
    def buy(self, open_date: datetime, buy_price: float, current_cash: float) -> tuple[Trade, float]:
        # Update the min/max price attributes
        if self.min_buy_price is None or buy_price < self.min_buy_price:
            self.min_buy_price = buy_price

        if self.max_buy_price is None or buy_price > self.max_buy_price:
            self.max_buy_price = buy_price

        # Open a new trade
        current_trade = Trade(
            opened=open_date,
            cash=current_cash,
            buy_price=buy_price,
            commission_pct=self.commission
        )
        self.num_buys += 1
        self.num_trades += 1
        current_cash = 0
        return current_trade, current_cash
    
    def complete(self, end_price: float, end_date: datetime, current_cash: float, current_trade: Optional[Trade] = None):
        self.end_price = end_price
        self.end_date = end_date
        self.duration = self.end_date - self.start_date
        self.duration_seconds = self.duration.total_seconds()
        self.buy_hold_return = (self.end_price - self.start_price) / self.start_price
        self.buy_hold_value = self.initial_cash * (1 + self.buy_hold_return)
        self.buy_hold_return *= 100

        if self.close_end_position and current_trade is not None:
            current_cash = self.sell(
                trade=current_trade,
                close_date=self.end_date,
                sell_price=self.end_price,
                is_end_position=True
            )
            current_trade = None
        
        # Set exposures as timedelta
        self.exposure_pct = (self.exposure / self.duration_seconds) * 100

        self.exposure = timedelta(seconds=self.exposure)
        if self.min_exposure is not None:
            self.min_exposure = timedelta(seconds=self.min_exposure)
        if self.max_exposure is not None:
            self.max_exposure = timedelta(seconds=self.max_exposure)
        
        # Get the final cash, shares, and value, depending on if there is a current trade
        if current_trade is not None:
            self.trades.append(current_trade)
            self.final_cash = 0
            self.final_shares = current_trade.shares
            self.final_value = self.final_shares * self.end_price * self.pct(self.commission, subtract=True)
        else:
            self.final_cash = current_cash
            self.final_shares = 0
            self.final_value = current_cash

        # Create the trades data
        for trade in self.trades:
            for key in self._trades_data:
                self._trades_data[key].append(getattr(trade, key))
        self.trades_data = pd.DataFrame(self._trades_data)

        winning_trades_df = self.trades_data.loc[self.trades_data['pnl'] > 0]
        losing_trades_df = self.trades_data.loc[self.trades_data['pnl'] < 0]
        self.avg_win = winning_trades_df['pnl'].mean()
        self.avg_win_pct = (winning_trades_df['pnl'] / winning_trades_df['cash'] * 100).mean()
        self.avg_loss = losing_trades_df['pnl'].mean()
        self.avg_loss_pct = (losing_trades_df['pnl'] / losing_trades_df['cash'] * 100).mean()
        self.sells_per_day = self.num_sells / (self.duration_seconds / 86_400)

        # Calculate the win percentage
        if self.num_sells > 0:
            self.win_pct = (self.num_wins / self.num_sells) * 100
            self.avg_win_weighted = (self.avg_win * self.num_wins) / self.num_sells
            self.avg_loss_weighted = (self.avg_loss * self.num_losses) / self.num_sells
        self.expected_value = self.avg_win_weighted + self.avg_loss_weighted

        # Calculate the return percentages
        self.return_pct = (self.final_value - self.initial_cash) / self.initial_cash
        self.apy = ((1 + self.return_pct) ** ((365 * 86_400) / self.duration_seconds)) - 1
        if self.apy < -1:
            self.apy = -1
        self.apy_value = self.initial_cash * (1 + self.apy)
        self.return_pct *= 100
        self.apy *= 100

    def sell(
        self, 
        trade: Trade, 
        close_date: datetime, 
        sell_price: float,
        is_stop_loss: bool = False,
        is_take_profit: bool = False,
        is_end_position: bool = False
    ) -> float:
        # Update the min/max price attributes
        if self.min_sell_price is None or sell_price < self.min_sell_price:
            self.min_sell_price = sell_price

        if self.max_sell_price is None or sell_price > self.max_sell_price:
            self.max_sell_price = sell_price

        # Close out the current trade
        trade.close(
            closed=close_date,
            sell_price=sell_price,
            is_stop_loss=is_stop_loss,
            is_take_profit=is_take_profit,
            is_end_position=is_end_position
        )
        current_cash = trade.cash
        self.num_sells += 1
        self.num_trades += 1

        # Update the min/max cash
        if current_cash < self.min_cash:
            self.min_cash = current_cash

        if current_cash > self.max_cash:
            self.max_cash = current_cash

        # Update wins/losses/pushes
        if trade.pnl == 0:
            self.num_pushes += 1
        elif trade.pnl > 0:
            self.num_wins += 1
            if trade.pnl > self.max_win:
                self.max_win = trade.pnl
            if trade.pnl_pct > self.max_win_pct:
                self.max_win_pct = trade.pnl_pct
        else:
            self.num_losses += 1
            if trade.pnl < self.max_loss:
                self.max_loss = trade.pnl
            if trade.pnl_pct < self.max_loss_pct:
                self.max_loss_pct = trade.pnl_pct

        # Update the exposure
        trade_duration_seconds = trade.duration.total_seconds()
        self.exposure += trade_duration_seconds
        if self.min_exposure is None or trade_duration_seconds < self.min_exposure:
            self.min_exposure = trade_duration_seconds

        if self.max_exposure is None or trade_duration_seconds > self.max_exposure:
            self.max_exposure = trade_duration_seconds

        self.trades.append(trade)     
        self.commission_paid += trade.commission       
        return current_cash
    
    def evaluate(
            self, 
            at_price: float, 
            at_date: datetime, 
            buy_signal: float, 
            sell_signal: float, 
            current_cash: float,
            current_trade: Optional[Trade] = None
        ) -> tuple[Trade, float]:        
        ###########################################################        
        def calc_pnl(shares: float, buy_price: float, sell_price: float) -> float:
            gross = shares * sell_price
            net = gross * self.pct(self.commission)
            return net - (shares * buy_price)
        ###########################################################
        # If current trade is None, then we are looking for a buy signal
        if current_trade is None:     
            # Buy Signal      
            if not isna(buy_signal) and current_cash > 0:
                current_trade, current_cash = self.buy(open_date=at_date, buy_price=at_price, current_cash=current_cash)
        
        # If current trade is not None then check sell signal as well as stop loss and take profit
        elif current_trade is not None:
            # Update the min/max price attributes for the open Trade
            current_trade.update_min_max_price(at_price)
            # Sell Signal
            if (
                not isna(sell_signal) 
                and current_trade.shares > 0 
                and not (self.restrict_sell_below_buy and at_price < current_trade.buy_price)
                and not (self.restrict_non_profitable and calc_pnl(current_trade.shares, current_trade.buy_price, at_price) < 0)
            ):
                current_cash = self.sell(
                    trade=current_trade,
                    close_date=at_date,
                    sell_price=at_price
                )
                current_trade = None

            # Stop Loss
            elif self.stop_loss is not None and at_price <= current_trade.buy_price * self.pct(self.stop_loss):
                current_cash = self.sell(
                    trade=current_trade,
                    close_date=at_date,
                    sell_price=at_price,
                    is_stop_loss=True
                )
                current_trade = None

            # Take Profit
            elif self.take_profit is not None and at_price >= current_trade.buy_price * self.pct(self.take_profit, subtract=False):
                current_cash = self.sell(
                    trade=current_trade,
                    close_date=at_date,
                    sell_price=at_price,
                    is_take_profit=True
                )
                current_trade = None
        return current_trade, current_cash
    
    @staticmethod    
    def pct(value: float, subtract: bool = True) -> float:
        if subtract:
            return (100 - value) / 100
        else:
            return (100 + value) / 100