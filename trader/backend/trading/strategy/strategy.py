import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from bokeh.plotting import figure, show
from collections.abc import Iterable
from matplotlib.dates import DateFormatter
from matplotlib.ticker import StrMethodFormatter

from .backtest import *
from .candles import Candles



class Strategy(ABC, object):
    """
    Abstract Base Class for Strategies
    """
    _wait_reprepare = True

    def __new__(cls, *args, **kwargs):
        cls.buy_column = f'{cls.__name__.lower()}_buy'
        cls.sell_column = f'{cls.__name__.lower()}_sell'
        return super().__new__(cls)

    def __init__(
            self, 
            data: pd.DataFrame, 
            use_data_inplace: bool = False, 
            column_mappers: Optional[dict] = None, 
            clean_columns: bool = True,
            init_prepare: bool = True
        ):
        self.data = format_data(
            data=data, 
            inplace=use_data_inplace, 
            column_mappers=column_mappers,
            clean_columns=clean_columns,
            buy_column=self.buy_column,
            sell_column=self.sell_column
        )
        self.indicators = []  # A list of dicts with info on the indicators applied to strategy
        self.BT = None

        # Get paramters from the class, if any
        self.parameters = {}
        for key, value in self.__dict__.items():
            if isinstance(value, self.Parameter):
                self.parameters[key] = value

        if init_prepare:
            self.prepare()
        self._wait_reprepare = False
    
    def __setattr__(self, name, value):
        if self._is_parameter(name):
            super().__setattr__(name, self.Parameter(value))
            if hasattr(self, 'data'):  # Params may be updated befor super init so skip if data is not yet set
                self.parameters[name] = getattr(self, name)
                if not self._wait_reprepare:
                    self.reprepare()
        else:
            super().__setattr__(name, value)
    
    # Parameter Class ################################################################################################################################
    class Parameter(float):
        def __new__(cls, value: Union[int, float]):
            return super().__new__(cls, value)
        

    # Abstract Method - must overide for inherited classes ##########################################################################################
    @abstractmethod
    def prepare(self) -> None:
        """
        The prepare method must be overridden by the inheriting class, this is where the indicators are calculated
        for the strategy as well as triggering the buy and sell signals. The buy and sell signals should be allocated to
        the self.data DataFrame in the columns names from the class attributed of Strategy.buy_column 
        and Strategy.sell_column, respectively.

        buy_column = f'{self.__class__.__name__.lower()}_buy'\n
        sell_column = f'{self.__class__.__name__.lower()}_sell'

        In self.data the buy_column and sell_column are set to np.nan upon Strategy initialization. Buy and Sell signals should be set 
        to anything that would not be considered null or False, such as 1.

        Example for SMA (Simple Moving Average) Strategy:
        ``` python
        def prepare(self) -> None:
            self.short_period = 25
            self.long_period = 100
            self.data['short_sma'] = self.I(talib.SMA, self.data['close'], self.short_period)
            self.data['long_sma'] = self.I(talib.SMA, self.data['close'], self.long_period)
            
            self.data.loc[self.data['short_sma'] > self.data['long_sma'], self.buy_column] = 1
            self.data.loc[self.data['short_sma'] < self.data['long_sma'], self.sell_column] = 1
        ```
        """
        pass
    #################################################################################################################################################
    # Indicator Method ##############################################################################################################################
    def I(
        self, 
        func: Callable,
        *func_args,
        i_names: Optional[Union[str, list, tuple]] = None,
        **func_kwargs,
        # TODO: Include params for plotting indicators
    ) -> None:
        """
        This method will apply an indicator to the data DataFrame. The indicator will be calculated and stored in the data DataFrame
        as the "name" arg or, if "name" is omitted then the lower name of the indicator function. 
        The indicator function must return a single value or a list of values that can be set as a column in the data DataFrame.
        """
        i_names = i_names
        #######################################################################################
        def handle_multiple(data_list, i_names=i_names):  
            if i_names is None or not isinstance(i_names, (list, tuple)):
                i_names = []
            elif not isinstance(i_names, (list, tuple)):
                i_names = [f'{i_names}_{i}' for i in range(1, len(data_list) + 1)]

            for i, (data, i_name) in enumerate(itertools.zip_longest(data_list, i_names, fillvalue='**INDICATOR_NA**'), 1):
                if isinstance(data, str) and data == '**INDICATOR_NA**':
                    # Need to check isinstance first because if the data is a pd.Series, it will raise 
                    # a ValueError: The truth value of a Series is ambiguous.
                    continue
                if i_name == '**INDICATOR_NA**':
                    i_name = f'{func.__name__.lower()}_{i}'
                self.data[str(i_name)] = data
                self.indicators.append({
                    'strategy': self.__class__.__name__,
                    'name': str(i_name),
                })
        #######################################################################################
        func_data = func(
            *func_args, 
            **filter_callable_kwargs(
                func=func,
                passed_kwargs=func_kwargs,
            )
        )
        
        # Check if the indicator function returns a DataFrame
        if isinstance(func_data, pd.DataFrame):
            data_list = [func_data[col] for col in func_data.columns]
            # handle_multiple(data_list, i_names)
            handle_multiple(data_list)
        
        # Check if the indicator function returns multiple values
        # But also check that the length of the tuple is not (conspicuously) the same length as the data DataFrame
        elif isinstance(func_data, tuple) and len(func_data) != len(self.data):  
            # handle_multiple(func_data, i_names)   
            handle_multiple(func_data)     
        
        # Check if the indicator function returns and array function that is not the same length as the data DataFrame
        # This may be a list or a numpy array with the data objects contained within
        elif hasattr(func_data, '__len__') and len(func_data) != len(self.data):
            # handle_multiple(func_data, i_names)
            handle_multiple(func_data)
        
        # Otherwise, it is assumed that the indicator is a single value
        else:
            if i_names is None:
                i_names = func.__name__.lower()
            elif isinstance(i_names, (list, tuple)):
                i_names = i_names[0]
            self.data[str(i_names)] = func_data
            self.indicators.append({
                'strategy': self.__class__.__name__,
                'name': str(i_names),
            })

    # Logging Methods ###############################################################################################################################
    def _log(self, message: str, level: Literal['INFO', 'WARNING'] = 'INFO') -> None:
        prtcolor(text=message, color_code=36, prefix=f'[{self.__class__.__name__} {level.upper()}]', add_newline=False)
    
    def log_info(self, message: str) -> None:
        self._log(message, level='INFO')
    
    def log_warn(self, message: str) -> None:
        self._log(message, level='WARNING')
    
    # Private Methods ##############################################################################################################################
    def _get_threshold_values(self, threshold: Any) -> tuple:
        if threshold in self.data.columns:
            threshold_value = self.data[threshold]
            threshold_shift = self.data[threshold].shift(1)
        else:
            threshold_value = threshold
            threshold_shift = threshold

        return threshold_value, threshold_shift
    
    @classmethod
    def _is_parameter(cls, name: str) -> bool:
        if hasattr(cls, name):
            attr = getattr(cls, name)
            return isinstance(attr, cls.Parameter)
        else:
            return False

    # Public Methods ################################################################################################################################
    def add_row(
            self, 
            dt: datetime, 
            close: float, 
            open: Optional[float] = None, 
            high: Optional[float] = None, 
            low: Optional[float] = None,
            volume: Optional[float] = None,
        ) -> None:
        df = pd.DataFrame({
            'date': [dt],
            'open': [open],
            'high': [high],
            'low': [low],
            'close': [close],
            'volume': [volume]
        })
        self.add_data(data=df)
    
    def add_data(self, data: pd.DataFrame, column_mappers: Optional[dict] = None) -> None:
        formatted_data = format_data(
            data=data, 
            inplace=False, 
            column_mappers=column_mappers,
            clean_columns=True,
            buy_column=self.buy_column,
            sell_column=self.sell_column
        )
        for col in self.data.columns:
            if col not in formatted_data.columns:
                formatted_data[col] = np.nan
        
        self.data = pd.concat([self.data, formatted_data], axis=0)
        self.reprepare()

    def apply_candle(self, candle_name: str) -> None:
        if hasattr(Candles, candle_name):
            candle_func = getattr(Candles, candle_name)
            candle_series = candle_func(self.data)
        # Check if the candle_name is without the "cdl" prefix
        elif hasattr(Candles, f'cdl_{candle_name}'):
            candle_func = getattr(Candles, f'cdl_{candle_name}')
            candle_series = candle_func(self.data, name_with_cdl_prefix=True)
        else:
            raise ValueError(f'"{candle_name}" is not a valid candlestick pattern, within the Candles class')
        candle_series = candle_func(self.data)
        self.data[candle_series.name] = candle_series
        
    def apply_candles(self, candle_names: list, ignore_invalid: bool = True) -> None:
        for candle_name in candle_names:
            try:
                self.apply_candle(candle_name)
            except ValueError as e:
                if not ignore_invalid:
                    raise e
    
    def apply_all_candles(self) -> None:
        for key, value in Candles.__dict__.items():
            if key.startswith('cdl_') and callable(value):
                self.apply_candle(key)
    
    def apply_crossover(self, crosser: str, threshold: Any, new: str, reset_new_if_exists: bool = True, subset_indexes: Optional[pd.Index] = None) -> None:
        """
        This method will apply a crossover signal to the data DataFrame. The crossover signal is when the crosser-column crosses over
        the threshold, so if the current crosser-value is greater than the threshold AND the previous crosser-value
        was less than or equal to the previous threshold.\n\n

        The "threshold" can be any value (that can be compared) or it can be a column value in the data DataFrame.
        The "threshold" will be checked to see if it is a column-value in the data DataFrame, if it is not a column in 
        the data DataFrame, then it will be assumed to be a value.\n\n

        The "new" column will be set to 1 when the crosser crosses over the threshold. The "new" column is created or overwritten
        each time this is called.\n\n

        :params
        crosser: str - The column name of the crosser value
        threshold: Any - The value or column name of the threshold
        new: str - The column name to set the crossunder signal to
        reset_new_if_exists: bool - If True, the new column will be reset to np.nan if it already exists in the data DataFrame
        subset_indexes: pd.Index - A subset of indexes to apply the crossover signal to
        """
        if (reset_new_if_exists and new in self.data.columns) or new not in self.data.columns:
            self.data[new] = np.nan

        t_value, t_shift = self._get_threshold_values(threshold)
        if subset_indexes is not None:
            self.data.loc[
                (   
                    self.data.index.isin(subset_indexes)
                    & (self.data[crosser] > t_value) 
                    & (self.data[crosser].shift(1) <= t_shift)
                ),
                new
            ] = 1
        else:
            self.data.loc[
                (
                    (self.data[crosser] > t_value) 
                    & (self.data[crosser].shift(1) <= t_shift)
                ),
                new
            ] = 1

    def apply_crossunder(self, crosser: str, threshold: str, new: str, reset_new_if_exists: bool = True, subset_indexes: Optional[pd.Index] = None):
        """
        This method will apply a crossunder signal to the data DataFrame. The crossunder signal is when the crosser-column crosses under
        the threshold-column, so if the current crosser-value is less than the threshold-value AND the previous crosser-value
        was greater than or equal to the previous threshold-value.\n\n

        The "threshold" can be any value (that can be compared) or it can be a column value in the data DataFrame.
        The "threshold" will be checked to see if it is a column-value in the data DataFrame, if it is not a column in 
        the data DataFrame, then it will be assumed to be a value.\n\n


        The "new" column will be set to 1 when the crosser crosses under the threshold. If the "new" column is 
        truly new (does not exist in the data DataFrame), then the default values will be np.nan.\n\n

        :params
        crosser: str - The column name of the crosser value
        threshold: str - The value or column name of the threshold
        new: str - The column name to set the crossunder signal to
        reset_new_if_exists: bool - If True, the new column will be reset to np.nan if it already exists in the data DataFrame
        subset_indexes: pd.Index - A subset of indexes to apply the crossunder signal to
        """
        if (reset_new_if_exists and new in self.data.columns) or new not in self.data.columns:
            self.data[new] = np.nan

        t_value, t_shift = self._get_threshold_values(threshold)
        if subset_indexes is not None:
            self.data.loc[
                (
                    self.data.index.isin(subset_indexes)
                    & (self.data[crosser] < t_value) 
                    & (self.data[crosser].shift(1) >= t_shift)
                ),
                new
            ] = 1
        else:
            self.data.loc[
                (
                    (self.data[crosser] < t_value)
                    & (self.data[crosser].shift(1) >= t_shift)
                ),
                new
            ] = 1
    

    def apply_profit_or_stop_loss(self, commission: float = 0, stop_loss: float = 2, min_profit: float = 0) -> None:
        """
        This method can be used to apply sell signals, after buy signals are established,
        which will sell only when there is a guaranteed profit or the current value drops below the
        stop loss threshold. If the "min_profit" argument is not None, the the profit must be above that
        percentage i.e. current_value > bought_value * (1 + (min_profit / 100))
        """
        # Get the fake value data
        fake_cash = 1_000
        fake_shares = 0
        bought_price = None

        # Get the adjustment percentages
        commission_adj_pct = 1 - (commission / 100)
        stop_loss_adj_pct = 1 - (stop_loss / 100)
        min_profit_adj_pct = 1 + (min_profit / 100)

        # Iterate through the data
        for row in self.data.itertuples():
            buy_signal = getattr(row, self.buy_column)
            if not isna(buy_signal) and fake_cash > 0:
                # Do buy
                bought_price = row.close
                fake_shares = fake_cash / bought_price
                fake_cash = 0
            
            elif fake_shares > 0:
                current_price = row.close
                current_value = fake_shares * current_price * commission_adj_pct
                bought_value = fake_shares * bought_price
                if current_value > bought_value * min_profit_adj_pct or current_value <= bought_value * stop_loss_adj_pct:
                    # Do sell
                    fake_cash = current_value
                    fake_shares = 0
                    self.data.at[row.Index, self.sell_column] = 1

    def backtest(
        self,  
        cash: float,
        commission: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False,
        close_end_position: bool = True,
    ) -> BackTest:
        self.BT = BackTest(
            strategy=self,
            cash=cash,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit,
            restrict_sell_below_buy=restrict_sell_below_buy,
            restrict_non_profitable=restrict_non_profitable,
            close_end_position=close_end_position
        )
        self.BT.run()
        return self.BT
    
    def backtest_cls(
        cls,
        data: pd.DataFrame,
        cash: float,
        commission: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False,
        close_end_position: bool = True,
        use_data_inplace: bool = False,
    ) -> BackTest:
        instance = cls(data, use_data_inplace=use_data_inplace)
        return instance.backtest(
            cash=cash,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit,
            restrict_sell_below_buy=restrict_sell_below_buy,
            restrict_non_profitable=restrict_non_profitable,
            close_end_position=close_end_position
        )
    
    def monte_carlo(
        self,
        interval: T.Trade.INTERVAL,
        n_periods: int,
        use_data_inplace: bool = False,
        mc_type: T.Strategy.MONTE_CARLO = 'normal',
        show_simulations: bool = False,
        show_historical: bool = False,
        price_min_bound: Optional[float] = None,
        price_max_bound: Optional[float] = None,
        induce_volatility: bool = False,
        volatility_max_std: float = 0.75
    ) -> pd.DataFrame:
        return monte_carlo(
            data=self.data,
            interval=interval,
            n_periods=n_periods,
            use_data_inplace=use_data_inplace,
            mc_type=mc_type,
            show_simulations=show_simulations,
            show_historical=show_historical,
            price_min_bound=price_min_bound,
            price_max_bound=price_max_bound,
            induce_volatility=induce_volatility,
            volatility_max_std=volatility_max_std
        )
    
    @classmethod
    def monte_carlo_cls(
        cls,
        data: pd.DataFrame,
        interval: T.Trade.INTERVAL,
        n_periods: int,
        use_data_inplace: bool = False,
        mc_type: T.Strategy.MONTE_CARLO = 'normal',
        show_simulations: bool = False,
        show_historical: bool = False,
        price_min_bound: Optional[float] = None,
        price_max_bound: Optional[float] = None,
        induce_volatility: bool = False,
        volatility_max_std: float = 0.75
    ) -> pd.DataFrame:
        return monte_carlo(
            data=data,
            interval=interval,
            n_periods=n_periods,
            use_data_inplace=use_data_inplace,
            mc_type=mc_type,
            show_simulations=show_simulations,
            show_historical=show_historical,
            price_min_bound=price_min_bound,
            price_max_bound=price_max_bound,
            induce_volatility=induce_volatility,
            volatility_max_std=volatility_max_std
        )

    def final_value_with_n_sells(            
        self,  
        cash: float,
        commission: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False
    ) -> tuple[float, int]:
        """
        This is a quick way to evaulate a strategy without running a full backtest. This method will return the final value
        of the strategy after running through the data DataFrame. The final value is the cash value after the last row of the data
        DataFrame has been processed. This method will also take into account the stop_loss or take_profit values.
        """
        ###########################################################
        def pct(value: float, subtract: bool = True) -> float:
            if subtract:
                return (100 - value) / 100
            else:
                return (100 + value) / 100
        
        def calc_pnl(shares: float, buy_price: float, sell_price: float) -> float:
            gross = shares * sell_price
            net = gross * commission_adj_pct
            return net - (shares * buy_price)
        ###########################################################
        current_cash = float(cash)
        current_shares = 0
        last_buy_price = None
        commission_adj_pct = pct(commission)
        stop_loss_adj_pct = pct(stop_loss) if stop_loss is not None else None
        take_profit_adj_pct = pct(take_profit, subtract=False) if take_profit is not None else None

        # Only get the data that have either buy or sell signals
        # and shrink the size of the dataframe to only the necessary columns
        signal_data = self.data.loc[(
            (self.data[self.buy_column].notna()) 
            | (self.data[self.sell_column].notna())
        ), ['close', self.buy_column, self.sell_column]]

        n_sells = 0
        for row in signal_data.itertuples():
            current_price = row.close
            buy_signal = getattr(row, self.buy_column)
            sell_signal = getattr(row, self.sell_column)

            # Buy Signal
            if not isna(buy_signal) and current_cash > 0:
                current_shares = current_cash / current_price
                current_cash = 0
                last_buy_price = current_price
            
            # Sell Signal
            elif (
                not isna(sell_signal) 
                and current_shares > 0 
                and not (restrict_sell_below_buy and current_price < last_buy_price)
                and not (restrict_non_profitable and calc_pnl(current_shares, last_buy_price, current_price) < 0)
            ):
                current_cash = current_shares * current_price * commission_adj_pct
                current_shares = 0
                n_sells += 1
            
            # Check stop loss and take profit
            elif current_shares > 0:
                # We have an open position, so check if we need to close it according to the stop loss or take profit
                if stop_loss is not None and current_price <= last_buy_price * stop_loss_adj_pct:
                    current_cash = current_shares * current_price * commission_adj_pct
                    current_shares = 0
                    n_sells += 1

                elif take_profit is not None and current_price >= last_buy_price * take_profit_adj_pct:
                    current_cash = current_shares * current_price * commission_adj_pct
                    current_shares = 0
                    n_sells += 1
        
        if current_shares > 0:
            return current_shares * self.data['close'].iloc[-1] * commission_adj_pct, n_sells
        else:
            return current_cash, n_sells

    def final_value(            
        self,  
        cash: float,
        commission: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False
    ) -> float:
        """
        This is a quick way to evaulate a strategy without running a full backtest. This method will return the final value
        of the strategy after running through the data DataFrame. The final value is the cash value after the last row of the data
        DataFrame has been processed. This method will also take into account the stop_loss or take_profit values.
        """
        return self.final_value_with_n_sells(
            cash=cash,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit,
            restrict_sell_below_buy=restrict_sell_below_buy,
            restrict_non_profitable=restrict_non_profitable
        )[0]
    
    def parameter_optimize(
        self, 
        cash: float,
        commission: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False,
        top_n: Optional[int] = 10,
        exclude_no_sells: bool = True,
        **parameter_ranges
    ) -> pd.DataFrame:
        """
        This method will optimize the parameters of a strategy. The parameter_ranges should be a dictionary with the keys as the 
        parameter names and the values as a tuple of the range of values to optimize. The method will return a dictionary with the 
        best parameters and the final value of the strategy with those parameters.
        """
        original_params = ((key, value) for key, value in self.parameters.items())
        parameters = {}
        for key, value in parameter_ranges.items():
            if self._is_parameter(key):
                if isinstance(value, Iterable):
                    parameters[key] = value
                else:
                    parameters[key] = [value]
        if len(parameters) == 0:
            raise ValueError('No parameters to optimize, either you have not supplied any parameters or the class does not have any defined Parameters')
        
        # itertools.product will return a tuple of all the combinations of the parameters
        # these will be ordered in the same order as the parameters dictionary
        parameter_keys_ordered = list(parameters.keys())
        all_combos = list(itertools.product(*parameters.values()))
        all_combos = [{key: val for key, val in zip(parameter_keys_ordered, combo)} for combo in all_combos]
        len_all_combos = len(all_combos)
        best_fv_combos = []

        try:
            with TimerInline('Optimizing Parameters Loop', print_starting=False) as ti:
                for i, combo in enumerate(all_combos):
                    ti.progress_bar(start_iter=0, at_iter=i, total_length=len_all_combos)
                    # Update the parameters
                    self.update_parameters(**combo)

                    # Use the final_value() method, because it is much faster 
                    # than a full backtest, once the best combo(s) are found then we will run
                    # full backtests for those combos to get the full details
                    final_value, n_sells = self.final_value_with_n_sells(
                        cash=cash,
                        commission=commission,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        restrict_sell_below_buy=restrict_sell_below_buy,
                        restrict_non_profitable=restrict_non_profitable
                    )
                    if n_sells == 0 and exclude_no_sells:
                        continue
                    best_fv_combos.append(dict(
                        final_value=final_value, 
                        buy_signals=self.num_buys,
                        sell_signals=self.num_sells,
                        **combo
                    ))
                    
        except KeyboardInterrupt:
            self.log_info('Optimization Stopped by User, returning best results so far...')
        
        except Exception as e:
            print('Problem with Optimization Loop')
            print(f'Current Parameters: {combo}')
            raise e
        
        # Get the best combo data into a DataFrame
        data = {}
        for best_combo in best_fv_combos:
            for key, value in best_combo.items():
                if key not in data:
                    data[key] = []
                data[key].append(value)
        df = pd.DataFrame(data)
        df.sort_values(by='final_value', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if top_n is not None:
            df = df[:top_n].copy(deep=True)

        # Reset the parameters to the original values
        self.update_parameters(**{key: value for key, value in original_params})
        return df    

    def plot_data(self) -> None:
        ###########################################################
        def random_color() -> tuple:
            rgb = [np.random.randint(0, 255) for _ in range(3)]
            return tuple(rgb)
        ###########################################################
        """
        bokeh.layouts import row, column funcs
        This can be used for "show"ing multiple figures at once        

        plot_data sync_axis

        # Candlestick
        gain = self.data['close'] > self.data['open']
        loss = self.data['open'] > self.data['close']

        figure.segment(dateindex, high, dateindex, low)
        figure.vbar(dateindex[gain], <width>, open[gain], close[gain], fill_color='green', line_color='green', line_width=1)
        figure.vbar(dateindex[loss], <width>, open[loss], close[loss], fill_color='red', line_color='red', line_width=1)






        """
        fig = figure(
            title='Close Prices', 
            x_axis_label='Date', 
            y_axis_label='Price', 
            x_axis_type='datetime',
            width_policy='max',
            height_policy='max',
        )
        
        # Add the indicators if they correspond to price values
        for indicator in self.indicators:
            if indicator['name'] in self.data.columns:
                median_value = self.data[indicator['name']].median()
                min_close = self.data['close'].min()
                max_close = self.data['close'].max()
                if min_close <= median_value <= max_close:
                    use_name = indicator['name']
                    if 'original_name' in indicator:
                        use_name = indicator['original_name']
                    if use_name not in ['upper_fib', 'lower_fib']:  # These are ugly on the graph
                        fig.line(self.data.index, self.data[indicator['name']], legend_label=use_name, line_width=2, color=random_color())
        
        # Plot the close prices
        fig.line(self.data.index, self.data['close'], legend_label='Close Price', line_width=2, color='blue')



        buy_signals = self.data.loc[self.data[self.buy_column].notna(), 'close']
        sell_signals = self.data.loc[self.data[self.sell_column].notna(), 'close']
        fig.scatter(buy_signals.index, buy_signals, legend_label='Buy Signal', color='green', marker='triangle', size=10)
        fig.scatter(sell_signals.index, sell_signals, legend_label='Sell Signal', color='red', marker='inverted_triangle', size=10)
        show(fig)

    def random_remove_signals(self, signal_type: Literal['buy', 'sell'], removal_percent: float = 25) -> None:
        """
        This method will randomly remove a percentage of the buy or sell signals from the data DataFrame.
        """
        if signal_type == 'buy':
            signal_column = self.buy_column
        elif signal_type == 'sell':
            signal_column = self.sell_column
        else:
            raise ValueError('signal_type must be either "buy" or "sell"')
        signals = self.data.loc[self.data[signal_column].notna(), signal_column]        
        n_signals = len(signals)
        n_remove = int(n_signals * (removal_percent / 100))
        remove_indexes = np.random.choice(signals.index, size=n_remove, replace=False)
        self.data.loc[remove_indexes, signal_column] = np.nan
    
    def reprepare(self) -> None:
        self.data[self.buy_column] = np.nan
        self.data[self.sell_column] = np.nan
        self.prepare()

    def update_parameters(self, **parameters) -> None:
        # Turn on _wait_reprepare to prevent repreparing the strategy while updating the parameters
        # which would happen in __setattr__
        self._wait_reprepare = True
        for key, value in parameters.items():
            if self._is_parameter(key):
                setattr(self, key, value)
        self._wait_reprepare = False
        self.reprepare()

    
    # Properties ###################################################################################################################################
    @property
    def num_buys(self) -> int:
        return len(self.data.loc[self.data[self.buy_column].notna()])
    
    @property
    def num_sells(self) -> int:
        return len(self.data.loc[self.data[self.sell_column].notna()])
    





