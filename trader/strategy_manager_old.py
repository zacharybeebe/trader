import atexit
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sqlite3
import ta
import yfinance as yf

from multiprocessing import Process
from threading import Thread

from datetime import datetime, timedelta
from dbpd import SQLite
from typing import Optional, Tuple

from .utils.utils import filter_callable_kwargs, get_default_kwargs, nan, timer


class StrategyManager:
    DATABASE = os.path.join(os.path.dirname(__file__), 'ticker_data.db')

    STRATEGY_PLOT_DEFS = {
        'bollinger': {
            'plot': {
                'bollinger_upper_band': dict(label='Upper Band', color='orange', linestyle='--'),
                'bollinger_lower_band': dict(label='Lower Band', color='pink', linestyle='--')
            }
        },

        'ema': {
            'plot': {
                'ema_short_ema': dict(label='Short EMA', color='orange', linestyle='--'),
                'ema_long_ema': dict(label='Long EMA', color='pink', linestyle='--')
            }
        },

        'fib_ret': {
            'plot': {
                'fib_ret_low_level': dict(label='Low Level', color='orange', linestyle='--'),
                'fib_ret_high_level': dict(label='High Level', color='pink', linestyle='--'),
                'low': dict(label='Low', color='orange', linestyle='--', lw=0.25),
                'high': dict(label='High', color='pink', linestyle='--', lw=0.25),
                'fib_ret_price_range': dict(label='Price Range', color='white', linestyle='-.', lw=0.25)
            }
        },

        'ichimoku_cloud': {
            'plot': {
                'ichimoku_cloud_tenkan_sen': dict(label='Tenkan Sen (Conversion Line)', color='orange', linestyle='--'),
                'ichimoku_cloud_kijun_sen': dict(label='Kijun Sen (Base Line)', color='pink', linestyle='--'),
                'ichimoku_cloud_kumo': dict(label='Kumo (Cloud)', color='white', linestyle='-.', lw=0.75)
            }
        },

        'macd': {
            'plot': {
                'macd_macd': dict(label='MACD', color='orange', linestyle='--'),
                'macd_signal_line': dict(label='Signal Line', color='pink', linestyle='--'),
                'macd_short_ema': dict(label='Short EMA', color='orange', linestyle='--', lw=0.25),
                'macd_long_ema': dict(label='Long EMA', color='pink', linestyle='--', lw=0.25)
            }
        },

        'momentum': {
            'plot': {
                'momentum_d_roc': dict(label='Display ROC (Rate of Change)', color='white', linestyle='-.', lw=0.25),
                'momentum_d_buy_thresh': dict(label='Display Buy Threshold', color='orange', linestyle='--'),
                'momentum_d_sell_thresh': dict(label='Display Sell Threshold', color='pink', linestyle='--')
            }
        },

        'price_breakout': {
            'plot': {
                'price_breakout_rolling_high': dict(label='Rolling High', color='orange', linestyle='--'),
                'price_breakout_rolling_low': dict(label='Rolling Low', color='pink', linestyle='--')
            }
        },

        'rsi': {
            'plot': {
                'rsi_rsi': dict(label='RSI', color='white', linestyle='-.', lw=0.25),
                'rsi_overbought_thresh': dict(label='Overbought Threshold', color='orange', linestyle='--'),
                'rsi_oversold_thresh': dict(label='Oversold Threshold', color='pink', linestyle='--')
            }
        },

        'rsi_dynamic': {
            'plot': {
                'rsi_dynamic_rsi': dict(label='RSI', color='white', linestyle='-.', lw=0.25),
                'rsi_dynamic_avg_price': dict(label='Avg Price', color='green', linestyle='--', lw=0.75),
                'rsi_dynamic_overbought_thresh': dict(label='Overbought Threshold', color='orange', linestyle='--'),
                'rsi_dynamic_oversold_thresh': dict(label='Oversold Threshold', color='pink', linestyle='--')
            }
        },

        'sma': {
            'plot': {
                'sma_short_sma': dict(label='Short SMA', color='orange', linestyle='--'),
                'sma_long_sma': dict(label='Long SMA', color='pink', linestyle='--')
            }
        },

        'sto_osc': {
            'plot': {
                'sto_osc_overbought_thresh': dict(label='Overbought Threshold', color='white', linestyle='-.', lw=0.5),
                'sto_osc_oversold_thresh': dict(label='Oversold Threshold', color='green', linestyle='-.', lw=0.5),
                'sto_osc_k': dict(label='%K', color='orange', linestyle='--'),
                'sto_osc_d': dict(label='%D', color='pink', linestyle='--')
            }
        },

        'vwap': {
            'plot': {
                'vwap_vwap': dict(label='VWAP', color='orange', linestyle='--')
            }
        },

    }

    TRADING_DAYS = [0, 1, 2, 3, 4]

    # Constructor Methods ############################################################################################################################
    def __init__(self, ticker: str, start_date: datetime, end_date: datetime, interval: str = '1d'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.table_name = f'{self.ticker}_{self.interval}'.lower()
        self.timedelta_years = (self.end_date - self.start_date).days / 365
        self.timedelta_years_int = int(round(self.timedelta_years))
        self.db = SQLite(filepath=self.DATABASE, show_description=False)

        self.data = self.get_stock_data()

        atexit.register(self._atexit_close)

    # Private Methods ################################################################################################################################
    def _atexit_close(self):
        try:
            print('Closing Database...')
            self.db.close()
        except:
            pass

    # Data Prep Methods ##############################################################################################################################
    def _create_table(self):
        sql = f"""
        CREATE TABLE {self.table_name} (
            [date] TEXT,
            [open] REAL,
            [high] REAL,
            [low] REAL,
            [close] REAL,
            [adj_close] REAL,
            [volume] REAL            
        );
        """
        self.db.query(sql=sql)
        self.db.commit()

    @staticmethod
    def _format_yf_data(data: pd.DataFrame) -> pd.DataFrame:
        # Convert Dates to datetimes
        if 'Datetime' in data.columns:
            data['Date'] = data['Datetime']
            data.drop('Datetime', axis=1, inplace=True)

        if 'Date' not in data.columns:
            data['Date'] = data.index

        # If Date column is string, convert to datetime
        if isinstance(data['Date'].iloc[0], str):
            data['Date'] = data['Date'].apply(pd.to_datetime)

        # Map columns
        mappers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for column in mappers:
            new_column = column.replace(' ', '_').lower()
            data[new_column] = data[column]
        data.drop(mappers, axis=1, inplace=True)
        data['date'] = data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return data

    def _get_all_data(self) -> Optional[pd.DataFrame]:
        if self._has_table():
            df = self.db.query(sql=f'SELECT * FROM {self.table_name} ORDER BY [date] ASC', show_head=False, warn_is_none=False)
            if df is not None:
                df['date'] = df['date'].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
            return df
        else:
            return None

    def _get_data_from_method_args(self, strategy_name: str, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if data is None:
            data = self.copy_data()

        if f'{strategy_name}_buy_signals' not in data.columns:
            data[f'{strategy_name}_buy_signals'] = nan()

        if f'{strategy_name}_sell_signals' not in data.columns:
            data[f'{strategy_name}_sell_signals'] = nan()
        return data

    def _has_table(self):
        return self.table_name in self.db.table_names(show_names=False)

    def _update_data(self, data: pd.DataFrame, ticker: str, interval: str = '1d'):
        table_name = f'{ticker}_{interval}'
        if not self._has_table():
            self._create_table()
            data.to_sql(name=table_name, con=self.db.db_conn, index=False)
        else:
            pass

    # Runs Static Methods ############################################################################################################################
    @staticmethod
    def _runs_process_positive_runs(positive_runs: dict, best: int = 5, print_only_top_best: bool = True) -> Tuple[dict, dict]:
        top_best = {}
        for k, v in positive_runs.items():
            if len(top_best) <= best:
                top_best[k] = v
            else:
                min_pct = min([vv['end_percent']] for vv in top_best.values())
                for kk, vv in top_best.items():
                    if vv['end_percent'] == min_pct:
                        top_best.pop(kk)
                        break

        print(f'\n\nPrinting Top {best} Runs')
        # Print Top 5
        for k, v in top_best.items():
            print(k)
            for kk, vv in v.items():
                print(f'\t{kk:20}{v}')

        if not print_only_top_best:
            print('\n\n\nPrinting All Runs')
            # Print all runs
            for k, v in positive_runs.items():
                print(k)
                for kk, vv in v.items():
                    print(f'\t{kk:20}{vv}')
        return top_best, positive_runs

    @timer
    def _runs_ema_or_sma(
            self,
            method: str,
            short_window_range: range,
            long_window_range: range,
            start_money: float,
            best: int = 5,
            print_only_top_best: bool = True

    ) -> Tuple[dict, dict]:
        ###############################################################################
        def threaded_backtest(positive_runs_dict, data_dict, at_short_window):
            for long_window in long_window_range:
                if at_short_window > long_window:
                    continue
                summary = self.backtest(
                    strategy_method_name=method,
                    start_money=start_money,
                    print_summary=False,
                    plot_strategy=False,
                    short_window=at_short_window,
                    long_window=long_window
                )
                if summary['end_percent'] > 0:
                    positive_runs_dict[(at_short_window, long_window)] = summary
                    data_dict['short_window'].append(at_short_window)
                    data_dict['long_window'].append(long_window)
                    data_dict['apy'].append(summary["end_apy"] * 100)
        ###############################################################################
        positive_runs = {}
        data = {'short_window': [], 'long_window': [], 'apy': []}
        threads = []
        for i in short_window_range:
            t = Thread(target=threaded_backtest, args=(positive_runs, data, i))
            t.start()
            threads.append(t)
            print(f'Completed Short Window: {i}')
        [t.join() for t in threads]

        df = pd.DataFrame(data)
        df.sort_values(by=['short_window', 'long_window'], axis=0, inplace=True)
        pivot = df.pivot(index='short_window', columns='long_window', values='apy')
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.show()
        return self._runs_process_positive_runs(positive_runs=positive_runs, best=best, print_only_top_best=print_only_top_best)

    # Utility Public Methods #########################################################################################################################
    def backtest(
            self,
            strategy_method_name: str,
            start_money: float = 10_000,
            skip_sell: bool = False,
            print_summary: bool = True,
            plot_strategy: bool = True,
            **strategy_method_kwargs
    ) -> dict:
        if not hasattr(self, strategy_method_name):
            raise Exception(f'{self.__class__.__name__} does not have the strategy method: "{strategy_method_name}"')

        # Get Method and kwargs
        strategy = getattr(self, strategy_method_name)
        strategy_method_kwargs = filter_callable_kwargs(func=strategy, passed_kwargs=strategy_method_kwargs)
        default_kwargs = get_default_kwargs(func=strategy)
        for k, v in default_kwargs.items():
            if k not in strategy_method_kwargs:
                strategy_method_kwargs[k] = v

        # Run the strategy method
        data = strategy(**strategy_method_kwargs)
        # print(data.to_string()[:1000])

        # Test the Buy and Sell signals
        current_money = float(start_money)  # Copy the start_money
        shares = 0

        buy_column = f'{strategy_method_name}_buy_signals'
        if skip_sell:
            sell_column = buy_column
        else:
            sell_column = f'{strategy_method_name}_sell_signals'

        for idx in data.index:
            if not pd.isna(data[buy_column][idx]) and current_money > 0:
                shares = current_money / data['adj_close'][idx]
                current_money = 0

            elif not pd.isna(data[sell_column][idx]) and shares > 0:
                current_money = shares * data['adj_close'][idx]
                shares = 0

        if current_money > 0:
            end_money = current_money
            statement = f'Cash In Hand: ${current_money:,.2f}'
        else:  # shares > 0
            end_money = shares * data['adj_close'].iloc[-1]
            statement = f'Value of Current Shares: ${end_money:,.2f}'

        end_percent = (end_money - start_money) / start_money
        if self.timedelta_years == 0:
            end_apy = 0
        else:
            end_apy = ((1 + end_percent) ** (1 / self.timedelta_years)) - 1############################################################################################

        summary = {
            'start_money': start_money,
            'end_money': end_money,
            'statement': statement,
            'end_percent': end_percent,
            'end_apy': end_apy
        }

        if print_summary:
            for k, v in summary.items():
                print(f'{k:20}{v}')
            print()

        if plot_strategy:
            strategy_method_kwargs.pop('data')
            self.plot_strategy(
                data=data,
                summary_from_test=summary,
                strategy_method_name=strategy_method_name,
                **strategy_method_kwargs
            )
        return summary

    def backtest_many(
            self,
            strategies_and_params: dict,
            start_money: float = 10_000,
            buy_sell_at_any_signal: bool = True,
            buy_sell_at_stength: Optional[int] = None,
            buy_at_combo_strength: list = [],
            skip_sell: bool = False
    ) -> Tuple[dict, pd.DataFrame]:
        data = self.copy_data()
        for strategy in strategies_and_params:
            if hasattr(self, strategy):
                # Add default args if not present
                default_kwargs = get_default_kwargs(getattr(self, strategy))
                for k, v in default_kwargs.items():
                    if k not in strategies_and_params[strategy]:
                        strategies_and_params[strategy][k] = v
                strategies_and_params[strategy]['data'] = data

                # print(f'{strategy=}')
                # print(strategies_and_params[strategy])
                # print()
                method = getattr(self, strategy)
                data = method(**strategies_and_params[strategy])

        # Test the Buy and Sell signals
        current_money = float(start_money)  # Copy the start_money
        shares = 0

        if skip_sell:
            sell_prefix = 'buy'
        else:
            sell_prefix = 'sell'

        for idx in data.index:
            # Check buy signals
            if current_money > 0:
                if buy_sell_at_stength is not None:
                    strength = 0
                    for strategy in strategies_and_params:
                        if not pd.isna(data[f'{strategy}_buy_signals'][idx]):
                            strength += 1
                    if strength == buy_sell_at_stength:
                        shares = current_money / data['adj_close'][idx]
                        current_money = 0
                        print(f'BUY {strength=}')

                elif buy_at_combo_strength:
                    if all([True if not pd.isna(data[f'{strategy}_buy_signals'][idx]) else False for strategy in buy_at_combo_strength]):
                        shares = current_money / data['adj_close'][idx]
                        current_money = 0
                        print(f'BUY {buy_at_combo_strength=}')
                else:
                    for strategy in strategies_and_params:
                        if not pd.isna(data[f'{strategy}_buy_signals'][idx]):
                            shares = current_money / data['adj_close'][idx]
                            current_money = 0
                            break

            # Check sell signals
            elif shares > 0:
                if buy_sell_at_stength is not None:
                    strength = 0
                    for strategy in strategies_and_params:
                        if not pd.isna(data[f'{strategy}_{sell_prefix}_signals'][idx]):
                            strength += 1
                    if strength == buy_sell_at_stength:
                        current_money = shares * data['adj_close'][idx]
                        shares = 0
                        print(f'SELL {strength=}')

                elif buy_at_combo_strength:
                    if all([True if not pd.isna(data[f'{strategy}_{sell_prefix}_signals'][idx]) else False for strategy in buy_at_combo_strength]):
                        current_money = shares * data['adj_close'][idx]
                        shares = 0
                        print(f'SELL {buy_at_combo_strength=}')

                else:
                    for strategy in strategies_and_params:
                        if not pd.isna(data[f'{strategy}_{sell_prefix}_signals'][idx]):
                            current_money = shares * data['adj_close'][idx]
                            shares = 0
                            break

        if current_money > 0:
            end_money = current_money
            statement = f'Cash In Hand: ${current_money:,.2f}'
        else:  # shares > 0
            end_money = shares * data['adj_close'].iloc[-1]
            statement = f'Value of Current Shares: ${end_money:,.2f}'

        end_percent = (end_money - start_money) / start_money
        if self.timedelta_years == 0:
            end_apy = 0
        else:
            end_apy = ((1 + end_percent) ** (1 / self.timedelta_years)) - 1  ############################################################################################

        summary = {
            'start_money': start_money,
            'end_money': end_money,
            'statement': statement,
            'end_percent': end_percent,
            'end_apy': end_apy
        }

        for k, v in summary.items():
            print(f'{k:20}{v}')

        return summary, data

    def copy_data(self) -> pd.DataFrame:
        data = self.data.copy(deep=True)
        return data

    def get_stock_data(self) -> pd.DataFrame:
        all_data = self._get_all_data()
        if all_data is None:
            # print('All data is None...')
            if not self._has_table():
                # print(f'\tCreating Table {self.table_name}...')
                self._create_table()

            # print(f'\tInserting Fresh Data Into Table {self.table_name}...')
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.interval)
            data = self._format_yf_data(data=data)
            data.to_sql(name=self.table_name, con=self.db.db_conn, if_exists='append', index=False)
        else:
            # print('All Data is NOT None...')
            min_date = all_data['date'].min().to_pydatetime() - timedelta(days=1)
            max_date = all_data['date'].max().to_pydatetime() + timedelta(days=1)
            if self.start_date < min_date and self.start_date.weekday() in self.TRADING_DAYS:
                # print('\tStart Date is less than Min Date...')
                data = yf.download(self.ticker, start=self.start_date, end=min_date, interval=self.interval)
                if not data.empty:
                    data = self._format_yf_data(data=data)
                    data.to_sql(name=self.table_name, con=self.db.db_conn, if_exists='append', index=False)

            if self.end_date > max_date and self.end_date.weekday() in self.TRADING_DAYS:
                # print('\tEnd Date is greater than Max Date...')
                data = yf.download(self.ticker, start=max_date, end=self.end_date, interval=self.interval)
                if not data.empty:
                    data = self._format_yf_data(data=data)
                    data.to_sql(name=self.table_name, con=self.db.db_conn, if_exists='append', index=False)

        all_data = self._get_all_data()
        data = all_data.loc[((all_data['date'] >= self.start_date) & (all_data['date'] <= self.end_date))]
        data = data.copy(deep=True)
        # print(data)
        return data

    # Individual Strategy Methods ####################################################################################################################
    def bollinger(
            self,
            window: int = 20,
            num_std_dev: float = 2,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Mean Reversion Strategy using Bollinger Bands.

        The strategy generates buy signals when the stock price is below the lower Bollinger Band
        and sell signals when the price is above the upper Bollinger Band.
        """
        strategy_name = 'bollinger'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate rolling mean and standard deviation
        rolling_mean = data['adj_close'].rolling(window=window).mean()
        rolling_std = data['adj_close'].rolling(window=window).std()

        # Calculate upper and lower Bollinger Bands
        data[f'{strategy_name}_upper_band'] = rolling_mean + (rolling_std * num_std_dev)
        data[f'{strategy_name}_lower_band'] = rolling_mean - (rolling_std * num_std_dev)

        holding = False
        for i, idx in enumerate(data.index, 1):
            if i < window:
                continue
            # Buy Signal
            if data['adj_close'][idx] < data[f'{strategy_name}_lower_band'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data['adj_close'][idx] > data[f'{strategy_name}_upper_band'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def ema(
            self,
            short_window: int = 12,
            long_window: float = 26,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        This strategy generates buy signals when the short-term EMA crosses above the long-term EMA
        and sell signals when the short-term EMA crosses below the long-term EMA
        """
        strategy_name = 'ema'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate short-term and long-term EMAs
        data[f'{strategy_name}_short_ema'] = data['adj_close'].ewm(span=short_window, adjust=False).mean()
        data[f'{strategy_name}_long_ema'] = data['adj_close'].ewm(span=long_window, adjust=False).mean()

        holding = False
        for i, idx in enumerate(data.index, 1):
            if i < long_window:
                continue
            # Buy Signal
            if data[f'{strategy_name}_short_ema'][idx] > data[f'{strategy_name}_long_ema'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data[f'{strategy_name}_short_ema'][idx] < data[f'{strategy_name}_long_ema'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def fib_ret(self, high_level: float = 0.618, low_level: float = 0.382, data: Optional[pd.DataFrame] = None):
        """
        This is the Fibonacci Retracement Strategy

        This strategy generates buy signals when the stock price touches or goes
        below the "high_level" retracement level and sell signals when the price touches or goes
        above the "low_level" retracement level
        """
        strategy_name = 'fib_ret'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate the price range for Fibonacci retracement
        price_range = data['high'].max() - data['low'].min()
        data[f'{strategy_name}_price_range'] = price_range

        # Calculate Fibonacci retracement levels
        data[f'{strategy_name}_low_level'] = data['high'].max() - low_level * price_range
        data[f'{strategy_name}_high_level'] = data['high'].max() - high_level * price_range

        holding = False
        for i, idx in enumerate(data.index, 1):
            # Buy Signal
            if data['low'][idx] <= data[f'{strategy_name}_high_level'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data['high'][idx] >= data[f'{strategy_name}_low_level'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def ichimoku_cloud(
            self,
            tenkan_window: int = 9,
            kijun_window: int = 26,
            senkou_span_b_window: int = 52,
            data: Optional[pd.DataFrame] = None
    ):
        """
        In this example, buy signals are generated when the Tenkan-sen crosses above the Kijun-sen
        and the price is above the cloud. Sell signals are generated when the Tenkan-sen crosses
        below the Kijun-sen or the price is below the cloud.
        """
        strategy_name = 'ichimoku_cloud'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Tenkan-sen (Conversion Line)
        data[f'{strategy_name}_tenkan_sen'] = (data['high'].rolling(window=tenkan_window).max() + data['low'].rolling(window=tenkan_window).min()) / 2

        # Kijun-sen (Base Line)
        data[f'{strategy_name}_kijun_sen'] = (data['high'].rolling(window=kijun_window).max() + data['low'].rolling(window=kijun_window).min()) / 2

        # Senkou Span A (Leading Span A)
        data[f'{strategy_name}_senkou_span_a'] = ((data[f'{strategy_name}_tenkan_sen'] + data[f'{strategy_name}_kijun_sen']) / 2).shift(kijun_window)

        # Senkou Span B (Leading Span B)
        data[f'{strategy_name}_senkou_span_b'] = ((data['high'].rolling(window=senkou_span_b_window).max() + data['low'].rolling(window=senkou_span_b_window).min()) / 2).shift(kijun_window)

        # Kumo (Cloud)
        data[f'{strategy_name}_kumo'] = data[f'{strategy_name}_senkou_span_a'] - data[f'{strategy_name}_senkou_span_b']

        holding = False
        for i, idx in enumerate(data.index, 1):
            # Buy Signal - Tenkan-sen crosses above Kijun-sen and price is above the cloud
            if data[f'{strategy_name}_tenkan_sen'][idx] > data[f'{strategy_name}_kijun_sen'][idx] and data['adj_close'][idx] > data[f'{strategy_name}_kumo'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal - Tenkan-sen crosses below Kijun-sen or price is below the cloud
            elif data[f'{strategy_name}_tenkan_sen'][idx] < data[f'{strategy_name}_kijun_sen'][idx] or data['adj_close'][idx] < data[f'{strategy_name}_kumo'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def macd(
            self,
            short_window: int = 12,
            long_window: int = 26,
            signal_window: int = 9,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        This strategy generates buy signals when the MACD line crosses above the Signal line and
        sell signals when the MACD line crosses below the Signal line
        """
        strategy_name = 'macd'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate short-term and long-term EMAs
        data[f'{strategy_name}_short_ema'] = data['adj_close'].ewm(span=short_window, adjust=False).mean()
        data[f'{strategy_name}_long_ema'] = data['adj_close'].ewm(span=long_window, adjust=False).mean()

        # Calculate MACD line
        data[f'{strategy_name}_macd'] = data[f'{strategy_name}_short_ema'] - data[f'{strategy_name}_long_ema']

        # Calculate Signal line (9-day EMA of MACD)
        data[f'{strategy_name}_signal_line'] = data[f'{strategy_name}_macd'].ewm(span=signal_window, adjust=False).mean()

        # Generate buy signals (MACD crosses above Signal line)
        data['BS1'] = (data[f'{strategy_name}_macd'] > data[f'{strategy_name}_signal_line']) & (data[f'{strategy_name}_macd'].shift(1) <= data[f'{strategy_name}_signal_line'].shift(1))

        # Generate sell signals (MACD crosses below Signal line)
        data['SS1'] = (data[f'{strategy_name}_macd'] < data[f'{strategy_name}_signal_line']) & (data[f'{strategy_name}_macd'].shift(1) >= data[f'{strategy_name}_signal_line'].shift(1))

        holding = False
        for i, idx in enumerate(data.index, 1):
            if i < long_window:
                continue
            # Buy Signal
            if data['BS1'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data['SS1'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]

        data.drop(['BS1', 'SS1'], axis=1, inplace=True)
        return data

    def momentum(
            self,
            roc_window: int = 14,
            buy_threshold: float = 1.02,
            sell_threshold: float = 0.98,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        The strategy generates buy signals when the Rate of Change (ROC) is above a certain
        threshold (indicating positive momentum) and sell signals when the ROC is
        below a certain threshold (indicating negative momentum).
        """
        strategy_name = 'momentum'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate Rate of Change (ROC)
        data[f'{strategy_name}_roc'] = 1 + data['adj_close'].pct_change(roc_window)

        data[f'{strategy_name}_d_roc'] = data[f'{strategy_name}_roc'] * data['adj_close'].mean()
        data[f'{strategy_name}_d_buy_thresh'] = buy_threshold * data['adj_close'].mean()
        data[f'{strategy_name}_d_sell_thresh'] = sell_threshold * data['adj_close'].mean()

        holding = False
        for i, idx in enumerate(data.index, 1):
            if i < roc_window:
                continue
            # Buy Signal
            if data[f'{strategy_name}_roc'][idx] > buy_threshold and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data[f'{strategy_name}_roc'][idx] < sell_threshold and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def price_breakout(
            self,
            window: int = 20,
            breakout_threshold: float = 1.02,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        This strategy identifies resistance levels by calculating the rolling high prices over a specified
        lookback period. Buy signals are generated when the closing price breaks above this resistance level,
        and sell signals are generated when the closing price breaks below the rolling low prices.
        """
        strategy_name = 'price_breakout'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate rolling high and low prices
        data[f'{strategy_name}_rolling_high'] = data['high'].rolling(window=window).max()
        data[f'{strategy_name}_rolling_low'] = data['low'].rolling(window=window).min()

        holding = False
        for i, idx in enumerate(data.index, 1):
            if i < window:
                continue
            # Buy Signal
            if data['adj_close'][idx] > data[f'{strategy_name}_rolling_high'][idx] * breakout_threshold and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data['adj_close'][idx] < data[f'{strategy_name}_rolling_low'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def rsi(
            self,
            rsi_window: int = 14,
            overbought_threshold: float = 70,
            oversold_threshold: float = 30,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        RSI - Relative Strength Index

        The thresholds are NOT prices, they are thresholds relative to the RSI
        The strategy generates buy signals when the RSI falls below the oversold threshold (e.g., 30)
        and sell signals when the RSI rises above the overbought threshold (e.g., 70)
        """
        strategy_name = 'rsi'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate RSI
        delta = data['adj_close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        data[f'{strategy_name}_overbought_thresh'] = overbought_threshold
        data[f'{strategy_name}_oversold_thresh'] = oversold_threshold

        data[f'{strategy_name}_avg_gain'] = gain.rolling(window=rsi_window, min_periods=1).mean()
        data[f'{strategy_name}_avg_loss'] = loss.rolling(window=rsi_window, min_periods=1).mean()

        data[f'{strategy_name}_rs'] = data[f'{strategy_name}_avg_gain'] / data[f'{strategy_name}_avg_loss']
        data[f'{strategy_name}_rsi'] = 100 - (100 / (1 + data[f'{strategy_name}_rs']))

        holding = False
        for i, idx in enumerate(data.index, 1):
            if i < rsi_window:
                continue
            # Buy Signal
            if data[f'{strategy_name}_rsi'][idx] < oversold_threshold and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data[f'{strategy_name}_rsi'][idx] > overbought_threshold and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def rsi_dynamic(
            self,
            rsi_window: int = 14,
            relative_mid: float = 50,
            overbought_std_mult: float = 2,
            oversold_std_mult: float = 2,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        RSI - Relative Strength Index
        The strategy generates buy signals when the RSI falls below the oversold deviation of the average price
        and sell signals when the RSI rises above the overbought deviation of the average price

        The overbought and oversold std multipliers are multipliers for the standard deviation of the average price. And then converted to
        the true RSI threshold proportional to the average price and the relative mid.
        """
        strategy_name = 'rsi_dynamic'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate RSI
        delta = data['adj_close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        adj_price_mean = data['adj_close'].mean()
        adj_price_std = data['adj_close'].std()
        overbought_price = adj_price_mean + (overbought_std_mult * adj_price_std)
        oversold_price = adj_price_mean - (oversold_std_mult * adj_price_std)

        data[f'{strategy_name}_overbought_thresh'] = (overbought_price * relative_mid) / adj_price_mean
        data[f'{strategy_name}_oversold_thresh'] = (oversold_price * relative_mid) / adj_price_mean

        data[f'{strategy_name}_avg_price'] = adj_price_mean
        data[f'{strategy_name}_avg_gain'] = gain.rolling(window=rsi_window, min_periods=1).mean()
        data[f'{strategy_name}_avg_loss'] = loss.rolling(window=rsi_window, min_periods=1).mean()

        data[f'{strategy_name}_rs'] = data[f'{strategy_name}_avg_gain'] / data[f'{strategy_name}_avg_loss']
        data[f'{strategy_name}_rsi'] = 100 - (100 / (1 + data[f'{strategy_name}_rs']))

        holding = False
        for i, idx in enumerate(data.index, 1):
            if i < rsi_window:
                continue
            # Buy Signal
            if data[f'{strategy_name}_rsi'][idx] < data[f'{strategy_name}_oversold_thresh'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data[f'{strategy_name}_rsi'][idx] > data[f'{strategy_name}_overbought_thresh'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def sma(
            self,
            short_window: int = 25,
            long_window: float = 100,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Simple moving average crossover strategy, two rolling means are calculated over their respective
        "windows" (a lookback period).

        If the short window average becomes greater than the long window average, that triggers a Buy signal.
        If the short window average becomes less than the long window average, that triggers a Sell signal.
        """
        strategy_name = 'sma'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate simple moving averages
        data[f'{strategy_name}_short_sma'] = data['adj_close'].rolling(window=short_window).mean()
        data[f'{strategy_name}_long_sma'] = data['adj_close'].rolling(window=long_window).mean()

        holding = False
        for i, idx in enumerate(data.index, 1):
            if i < long_window:
                continue
            # Buy Signal
            if data[f'{strategy_name}_short_sma'][idx] > data[f'{strategy_name}_long_sma'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data[f'{strategy_name}_short_sma'][idx] < data[f'{strategy_name}_long_sma'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    def sto_osc(
            self,
            k_window: int = 14,
            d_window: int = 3,
            overbought_threshold: float = 80,
            oversold_threshold: float = 20,
            data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        This is the Stochastic Oscillator strategy
        In this example, the strategy generates buy signals when %K crosses above %D and %K is below the
        oversold threshold. It generates sell signals when %K crosses below %D and %K is above the
        overbought threshold.
        """
        strategy_name = 'sto_osc'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        data[f'{strategy_name}_overbought_thresh'] = overbought_threshold
        data[f'{strategy_name}_oversold_thresh'] = oversold_threshold

        # Calculate sto_osc_k (%K) and sto_osc_d (%D)
        data[f'{strategy_name}_lowest_low'] = data['low'].rolling(window=k_window, min_periods=1).min()
        data[f'{strategy_name}_highest_high'] = data['high'].rolling(window=k_window, min_periods=1).max()

        data[f'{strategy_name}_k'] = ((data['adj_close'] - data[f'{strategy_name}_lowest_low']) / (data[f'{strategy_name}_highest_high'] - data[f'{strategy_name}_lowest_low'])) * 100
        data[f'{strategy_name}_d'] = data[f'{strategy_name}_k'].rolling(window=d_window, min_periods=1).mean()

        # Generate buy signals (sto_osc_k crosses above sto_osc_d and sto_osc_k is below oversold threshold)
        data['BS1'] = (data[f'{strategy_name}_k'] > data[f'{strategy_name}_d']) & (data[f'{strategy_name}_k'].shift(1) <= data[f'{strategy_name}_d'].shift(1)) & (data[f'{strategy_name}_k'] < oversold_threshold)

        # Generate sell signals (sto_osc_k crosses below sto_osc_d and sto_osc_k is above overbought threshold)
        data['SS1'] = (data[f'{strategy_name}_k'] < data[f'{strategy_name}_d']) & (data[f'{strategy_name}_k'].shift(1) >= data[f'{strategy_name}_d'].shift(1)) & (data[f'{strategy_name}_k'] > overbought_threshold)

        holding = False
        for i, idx in enumerate(data.index, 1):
            # Buy Signal
            if data['BS1'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data['SS1'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]

        data.drop(['BS1', 'SS1'], axis=1, inplace=True)
        return data

    def vwap(self, data: Optional[pd.DataFrame] = None):
        """
        VWAP - Volume-Weighted Average Price
        This strategy generates buy signals when the stock price is below the VWAP
        and sell signals when the price is above the VWAP
        """
        strategy_name = 'vwap'
        data = self._get_data_from_method_args(strategy_name=strategy_name, data=data)

        # Calculate VWAP
        data[f'{strategy_name}_dollar_volume'] = data['adj_close'] * data['volume']
        data[f'{strategy_name}_cumulative_dollar_volume'] = data[f'{strategy_name}_dollar_volume'].cumsum()
        data[f'{strategy_name}_cumulative_volume'] = data['volume'].cumsum()
        data[f'{strategy_name}_vwap'] = data[f'{strategy_name}_cumulative_dollar_volume'] / data[f'{strategy_name}_cumulative_volume']

        holding = False
        for i, idx in enumerate(data.index, 1):
            # Buy Signal
            if data['adj_close'][idx] < data[f'{strategy_name}_vwap'][idx] and not holding:
                holding = True
                data.at[idx, f'{strategy_name}_buy_signals'] = data['adj_close'][idx]

            # Sell Signal
            elif data['adj_close'][idx] > data[f'{strategy_name}_vwap'][idx] and holding:
                holding = False
                data.at[idx, f'{strategy_name}_sell_signals'] = data['adj_close'][idx]
        return data

    # Plotting Methods usually called by the test_strategy() method ##################################################################################
    def plot_strategy(self, data: pd.DataFrame, summary_from_test: dict, strategy_method_name: str, **strategy_method_kwargs) -> None:
        buy_signals_name = f'{strategy_method_name}_buy_signals'
        sell_signals_name = f'{strategy_method_name}_sell_signals'

        plt.style.use('dark_background')

        # Clear out plt
        plt.cla()

        # Format Labels
        tb = '        '
        strategy = ' '.join([i.capitalize() for i in strategy_method_name.split('_')])
        params = tb.join([f'{k}: {v}' for k, v in strategy_method_kwargs.items()])

        s_money = f"Start Money: \${summary_from_test['start_money']:,.2f}"
        e_value = f"Ending Value: \${summary_from_test['end_money']:,.2f}"
        g_pct = f"Gross Percent: {summary_from_test['end_percent'] * 100:.2f}%"
        apy = f"APY: {summary_from_test['end_apy'] * 100:.2f}%"
        statement = summary_from_test['statement'].replace('$', '\$')

        trading_days = len(data)  # Only trading days data will be in the dataframe
        trades_total = len(data.loc[(~(pd.isna(data[buy_signals_name])) | (~(pd.isna(data[sell_signals_name]))))]) * 2
        trades_per_day = round(trades_total / trading_days, 2)
        trades_per_year = round(trades_total / self.timedelta_years, 2)

        # Create Title
        plt.title(f"""
        {self.ticker} {self.timedelta_years} Years ---- {statement} ---- Avg Trades per Year: {trades_per_year} ---- Trades per Day: {trades_per_day}
        Strategy: {strategy} ---- Params: {params}
        {s_money}{tb}{e_value}{tb}{g_pct}{tb}{apy}
        """)

        # Plot method specific items
        for key in self.STRATEGY_PLOT_DEFS[strategy_method_name]:
            if key == 'plot':
                for data_column_name, plot_kwargs in self.STRATEGY_PLOT_DEFS[strategy_method_name][key].items():
                    plt.plot(data['date'], data[data_column_name], **plot_kwargs)
            elif key == 'scatter':
                pass

        # Plot shared items
        plt.plot(data['date'], data['adj_close'], label='Share Price', alpha=0.5)
        plt.scatter(x=data['date'], y=data[buy_signals_name], label='Buy Signal', marker='^', color='#00ff00', lw=3)
        plt.scatter(x=data['date'], y=data[sell_signals_name], label='Sell Signal', marker='v', color='#ff0000', lw=3)

        # Format axes
        plt.xlabel('Date')
        plt.ylabel('Stock Price')

        # Format Legend
        if data['adj_close'].iloc[-1] < data['adj_close'].iloc[0]:
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='upper left')

        # Show Plot
        plt.show()

    # Running Multiple Backtest Methods ##############################################################################################################
    def runs_bollinger(
            self,
            window_range: range,
            num_std_dev_range: range,
            start_money: float = 10_000,
            best: int = 5,
            print_only_top_best: bool = True
    ) -> Tuple[dict, dict]:
        positive_runs = {}
        for i in window_range:
            for j in num_std_dev_range:
                summary = self.backtest(
                    strategy_method_name='bollinger',
                    start_money=start_money,
                    print_summary=False,
                    plot_strategy=False,
                    window=i,
                    num_std_dev=j / 10
                )
                if summary['end_percent'] > 0:
                    positive_runs[(i, j)] = summary
            print(f'Completed Window: {i}')
        return self._runs_process_positive_runs(positive_runs=positive_runs, best=best, print_only_top_best=print_only_top_best)

    def runs_ema(
            self,
            short_window_range: range,
            long_window_range: range,
            start_money: float = 10_000,
            best: int = 5,
            print_only_top_best: bool = True,
            run_threaded: bool = True

    ) -> Tuple[dict, dict]:
        return self._runs_ema_or_sma(
            method='ema',
            short_window_range=short_window_range,
            long_window_range=long_window_range,
            start_money=start_money,
            best=best,
            print_only_top_best=print_only_top_best
        )

    def runs_macd(
            self,
            short_window_range: range = range(6, 19),
            long_window_range: range = range(13, 40),
            signal_window_range: range = range(5, 15),
            start_money: float = 10_000,
            best: int = 5,
            print_only_top_best: bool = True
    ) -> Tuple[dict, dict]:
        positive_runs = {}
        for i in short_window_range:
            for j in long_window_range:
                for k in signal_window_range:
                    summary = self.backtest(
                        strategy_method_name='macd',
                        start_money=start_money,
                        print_summary=False,
                        plot_strategy=False,
                        short_window=i,
                        long_window=j,
                        signal_window=k
                    )
                    if summary['end_percent'] > 0:
                        positive_runs[(i, j)] = summary
                    print(f'\t\tCompleted Signal Window: {k}')
                print(f'\tCompleted Long Window: {j}')
            print(f'Completed Short Window: {i}')
        return self._runs_process_positive_runs(positive_runs=positive_runs, best=best, print_only_top_best=print_only_top_best)

    def runs_rsi(
            self,
            overbought_threshold_range: range,
            oversold_threshold_range: range,
            start_money: float = 10_000,
            best: int = 5,
            print_only_top_best: bool = True
    ) -> Tuple[dict, dict]:
        positive_runs = {}
        for i in oversold_threshold_range:
            for j in overbought_threshold_range:
                # if i > j:
                #     continue
                summary = self.backtest(
                    strategy_method_name='rsi',
                    start_money=start_money,
                    print_summary=False,
                    plot_strategy=False,
                    short_window=i,
                    long_window=j
                )
                if summary['end_percent'] > 0:
                    positive_runs[(i, j)] = summary
            print(f'Completed Oversold Threshold Window: {i}')
        return self._runs_process_positive_runs(positive_runs=positive_runs, best=best, print_only_top_best=print_only_top_best)

    def runs_sma(
            self,
            short_window_range: range = range(4, 53, 2),
            long_window_range: range = range(12, 121, 4),
            start_money: float = 10_000,
            best: int = 5,
            print_only_top_best: bool = True,
            run_threaded: bool = True
    ) -> Tuple[dict, dict]:
        return self._runs_ema_or_sma(
            method='sma',
            short_window_range=short_window_range,
            long_window_range=long_window_range,
            start_money=start_money,
            best=best,
            print_only_top_best=print_only_top_best
        )




