import numpy as np
from trader.backend.data.engine import Engine
from trader.backend.trading.strategy.strategies import *
import itertools

if __name__ == '__main__':



    """
    crypto ['DOGE', 'BTC', 'ETH', 'LTC']  1m AND 5m"
    airlines ['UAL', 'DAL', 'AAL', 'LUV', 'ALK', 'BA']  ONLY 5m"

    LAST ALL = 01/03/2025 08:45
    """
    # Define the parameters
    ticker = 'BTC'
    interval = '5m'

    cash = 5_000
    commission = 100 - 99.54
    # commission = 0
    # start = datetime(
    #     year=2025, 
    #     month=1, 
    #     day=8, 
    #     hour=12
    # )
    # start = datetime(
    #     year=2024, 
    #     month=12, 
    #     day=13, 
    #     hour=12
    # )
    start = None
    # end = datetime(
    #     year=2024, 
    #     month=12, 
    #     day=20,
    #     hour=16
    # )
    end = None
    # end = datetime(
    #     year=2025, 
    #     month=1, 
    #     day=3,
    #     hour=8,
    #     minute=45
    # )

    eng = Engine(database_type='sqlite', engine_echo=False)
    trx = eng.start_transaction()
    # for ticker in ['DOGE', 'BTC', 'ETH', 'LTC']:
    # Get the candle data
    df = eng.data(
        ticker=ticker, 
        interval=interval,
        start=start,
        end=end,
        transaction=trx
    )
    print(df)
    print()
    trx.close()
    eng.close()

    # Strategies
    # strt = Combo(and_list=[TRIMA, Bollinger], or_list=[FibonacciRSI], data=df, column_mappers={'dt': 'date'})
    # strt = AllStrategiesOr(data=df, column_mappers={'dt': 'date'})
    # strt = KAMA(data=df, column_mappers={'dt': 'date'})
    # strt = KAMA(data=df, column_mappers={'dt': 'date'})    
    # strt = Combo(or_list=[TRIMA, MAMA], data=df, column_mappers={'dt': 'date'})
    # strt = FibonacciRSI(data=mcdf)
    # strt = FibonacciRSI(
    #     data=df, 
    #     column_mappers={'dt': 'date'},
    #     period=27,
    #     overbought=75,
    #     oversold=55
    # )
    # strt = FibonacciRSI(
    #     data=df, 
    #     column_mappers={'dt': 'date'},
    #     period=22,
    #     overbought=77,
    #     oversold=58
    # )
    # strt = Bollinger(df, column_mappers={'dt': 'date'})
    # strt = StochasticRSI(data=mcdf)
    # strt = MACD(data=mcdf)
    # strt = DogeLTCSlayer(data=df, column_mappers={'dt': 'date'})

    # strt = Combo(
    #     or_list=[FibonacciRSI, SMA],
    #     data=df,
    #     column_mappers={'dt': 'date'}
    # )

    # strt = FibSmaMod(data=df, column_mappers={'dt': 'date'})
    # strt = PSLMacd(data=df, column_mappers={'dt': 'date'})
    # strt = PSLSma(data=df, column_mappers={'dt': 'date'})
    # strt = PSLKama(data=df, column_mappers={'dt': 'date'})
    # strt = PSLFibSmaMod(data=df, column_mappers={'dt': 'date'})
    # strt = PSLDogeLTCSlayer(data=df, column_mappers={'dt': 'date'})

    # strt = TrimaMamaCross(data=df, column_mappers={'dt': 'date'})
    # strt = LinearReg(data=df, column_mappers={'dt': 'date'})
    # strt = LinRegEmaSmaCross(data=df, column_mappers={'dt': 'date'})

    strt = STDPeriods(
        data=df, 
        column_mappers={'dt': 'date'}, 
        period_len=200, 
        std_multiplier=0.05,
        trending_on=1,
        trend_divisor=2
    )

    # strt = Spikes(
    #     data=df, 
    #     column_mappers={'dt': 'date'},

    #     #BTC1m
    #     # upper_look_behind=3,
    #     # lower_look_behind=4,
    #     # upper_spike=0.8,
    #     # lower_spike=1.2,

    #     #DOGE1m
    #     upper_look_behind=3,
    #     lower_look_behind=2,
    #     upper_spike=0.99,
    #     lower_spike=0.67,

    #     # LTC1m
    #     # upper_look_behind=2,
    #     # lower_look_behind=3,
    #     # upper_spike=0.6,
    #     # lower_spike=1.2,

    #     # ETH1m
    #     # upper_look_behind=5,
    #     # lower_look_behind=5,
    #     # upper_spike=0.7,
    #     # lower_spike=1.1,

    #     # DOGE5m
    #     # upper_look_behind=2,
    #     # lower_look_behind=4,
    #     # upper_spike=3.9,
    #     # lower_spike=1.1,

    #     # DOGE5mDOWN  ***** VERY INSTERESTING *******
    #     # upper_look_behind=2,
    #     # lower_look_behind=2,
    #     # upper_spike=3.9,
    #     # lower_spike=3.8,

    #     # LTC5m 
    #     # upper_look_behind=4,
    #     # lower_look_behind=3,
    #     # upper_spike=3.6,
    #     # lower_spike=2.3,

    #     # ETH5m
    #     # upper_look_behind=2,
    #     # lower_look_behind=4,
    #     # upper_spike=2.4,
    #     # lower_spike=2.4,

    #     # BTC5m
    #     # upper_look_behind=2,
    #     # lower_look_behind=2,
    #     # upper_spike=2.7,
    #     # lower_spike=2.0,

    # )
    # strt.plot_data()

    # odf = strt.parameter_optimize(
    #     cash=cash,
    #     commission=commission,
    #     top_n=100,
    #     # stop_loss=2,
    #     # take_profit=10,
    #     # period=range(2, 201),
    #     # short_period=range(2, 102),
    #     # long_period=range(30, 130),
    #     period_len=range(200, 301),
    #     std_multiplier=[i / 20 for i in range(1, 21)],
    #     trending_on=[1, 0],
    #     trend_divisor=range(2, 11),
    #     # upper_look_behind=range(2, 6),
    #     # lower_look_behind=range(2, 6),
    #     # upper_spike=[i / 100 for i in range(1, 101)],
    #     # lower_spike=[i / 100 for i in range(1, 101)],
    #     # signal_period=range(2, 32),
    #     # overbought=range(60, 81),
    #     # oversold=range(40, 71),
    #     # period=range(20, 26),
    #     # overbought=range(80, 91),
    #     # oversold=range(50, 66),
    # )
    # print(odf.to_string())


    # # Optimization of the Combo strategy
    # optimized = strt.backtest_optimize(        
    #     cash=cash,
    #     commission=commission,
    #     # stop_loss=2,
    #     # take_profit=10,
    #     close_end_position=False,
    #     top_n=5,
    #     # parameter_ranges={
    #     #     'KAMA': {
    #     #         'period': range(12, 15)
    #     #     }
    #     # }
    # )
    # for obt, combo, params in optimized:
    #     print(f'\n\nBEST BACKTEST\n{combo=}')
    #     print(f'{params=}')
    #     # debug_print_dict(params)
    #     print(obt)
    #     print()
    
    # optimized_df = strt.parameter_optimize(        
    #     cash=cash,
    #     commission=commission,
    #     # stop_loss=2,
    #     # take_profit=10,
    #     top_n=5,        
    #     # restrict_non_profitable=True,


    #     # fast_limit=[ii / 10 for ii in range(2, 10)],
    #     # slow_limit=[ii / 100 for ii in range(2, 16)],
    #     lin_period=range(75, 125),
    #     sma_period=range(2, 53),
    #     ema_period=range(2, 53),
    #     # parameter_ranges={
    #     #     # 'ATR': {
    #     #     #     'period': range(13, 16),
    #     #     #     'multiplier': [i / 10 for i in range(18, 22)]
    #     #     # },
    #     #     # 'DEMA': {
    #     #     #     'short_period': range(11, 14),
    #     #     #     'long_period': range(25, 28)
    #     #     # },
    #     #     # 'EMA': {
    #     #     #     'short_period': range(19, 22),
    #     #     #     'long_period': range(49, 52)
    #     #     # },
    #     #     # 'SMA': {
    #     #     #     'short_period': range(17, 31),
    #     #     #     'long_period': range(110, 161)
    #     #     # },
    #     #     # 'FibonacciRSI': {
    #     #     #     'period': range(20, 23), # range(20, 41),
    #     #     #     'overbought': range(71, 74),
    #     #     #     'oversold': range(49, 52)
    #     #     # },
    #     #     # 'KAMA': {
    #     #     #     # 'period': range(9, 19),
    #     #     #     'kama_period': range(2, 51)
    #     #     # },
    #     #     # 'EMA': {
    #     #     #     'dema_period': range(2, 101),
    #     #     # }
    #     #     # 'TRIMA': {
    #     #     #     'period': range(13, 16)
    #     #     # }
    #     #     # 'StochasticRSI': {
    #     #     #     'period': range(9, 19),
    #     #     #     'k_period': range(3, 9),
    #     #     #     'd_period': range(2, 9),
    #     #     #     'overbought': [i / 100 for i in range(75, 85)],
    #     #     #     'oversold': [i / 100 for i in range(15, 25)]
    #     #     # },
    #     # }
    # )
    # print(optimized_df)

    # for i in optimized_df.to_dict(orient='records'):
    #     debug_print_dict(i)
    # strt.update_parameters(kama_period=optimized_df['kama_period'].iloc[0], dema_period=optimized_df['dema_period'].iloc[0])

    # Backtest the strategy
    buys = strt.data[strt.data[strt.buy_column].notna()]
    sells = strt.data[strt.data[strt.sell_column].notna()]
    print('\nBUY SIGNALS')
    print(buys)
    print('\nSELL SIGNALS')
    print(sells)
    print()
    bt = strt.backtest(
        cash=cash,
        commission=commission,
        # stop_loss=2,
        # take_profit=10,
        # restrict_sell_below_buy=True,
        # restrict_non_profitable=True,
        close_end_position=False
    )
    print(bt)
    print()
    print(bt.trades_data)
    print()
    # print(bt.trades_data.loc[bt.trades_data['pnl'] > 0, ['cash', 'shares', 'buy_price', 'sell_price', 'commission', 'pnl', 'duration']])
    print(bt.trades_data[['cash', 'shares', 'buy_price', 'sell_price', 'commission', 'pnl', 'duration']])
    print()
    debug_print_dict(strt.parameters)