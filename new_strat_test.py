


if __name__ == '__main__':
    from itertools import zip_longest
    import numpy as np
    import talib
    from trader.backend.data.engine import (
        Engine, 
        pd, 
        isna, 
        datetime, 
        datetime_parse, 
        get_dt_at_n_periods,
        n_periods_from_timedelta,
        timezone, 
        TimerInline, 
        timedelta, 
        generate_uuid, 
        yf, 
        utc_to_pst, 
        debug_print_dict, 
        unique_file_name,
        T
    )

    from trader.backend.trading.strategy.strategies import *



    d = {
        'p': [100 + random.random() * 10 for _ in range(10)],
    }
    df = pd.DataFrame(d)
    df['p_change'] = df['p'].pct_change() + 1
    df['buy'] = np.nan
    df['sell'] = np.nan
    df.loc[df['p_change'] > 1.01, 'sell'] = 1
    df.loc[df['p_change'] < 0.995, 'sell'] = -1
    print(df.to_string())





    
    # "tickers = [DOGE, BTC, ETH, LTC]"
    # " airline tickers = [UAL, DAL, AAL, LUV, ALK]"

    # t = 'DOGE'
    # i = '1m'

    # # # n_periods_behind = 6527
    # n_periods_behind = n_periods_from_timedelta(td=timedelta(hours=int(3.5*24)), interval=i)  # 5
    # # n_periods_behind = n_periods_from_timedelta(td=timedelta(days=38), interval=i)  # 5
    # print(f'{n_periods_behind=}')

    # # # TODO: Below is a patch knowing the interval, but make a dynamic way to find the satrt date based off of number of periods and an 'X' multipler of look-behind periods
    
    # end = None
    # if i == '5m':
    #     start = datetime.now(tz=timezone.utc) - timedelta(minutes=5 * n_periods_behind)
    # else:
    #     start = None
    # print(f'{start=}')

    # e = Engine(database_type='sqlite', engine_echo=False)#, drop_all_tables_on_init=True, create_all_tables_on_init=True)
    # trx = e.start_transaction()
    
    # with TimerInline('get ticker'):
    #     # df = e.data(
    #     #     ticker=t, 
    #     #     interval=i, 
    #     #     # start=datetime(year=2024, month=12, day=21, hour=9, minute=0, second=0, tzinfo=timezone.utc), 
    #     #     # end=datetime(year=2024, month=12, day=24, hour=0, minute=0, second=0, tzinfo=timezone.utc), 
    #     #     transaction=trx
    #     # )
    #     df = e.data(
    #         ticker=t, 
    #         interval=i, 
    #         start=datetime(year=2024, month=12, day=23, hour=23, minute=18, second=0), 
    #         # end=datetime(year=2024, month=12, day=23, hour=0, minute=0, second=0), 
    #         end=None, 
    #         transaction=trx
    #     )
    #     actual_start = df['dt'].iloc[0]
    #     print(df[-5:].to_string())
    #     print(f'{actual_start=}')
    
    # print(f'\nstart_price = ${df["close"].iloc[0]:,.6f} @ {df["dt"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")}')
    # print(f'end_price = ${df["close"].iloc[-1]:,.6f} @ {df["dt"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")}\n')


    # trx.close()    
    # e.close()
    # # n_periods = int((60 *24 * 365) / 5)
    # # n_periods = 10_000
    # n_periods = n_periods_from_timedelta(td=timedelta(days=7), interval=i)  # 5
    # print(f'{n_periods=}')

    # # for _ in range(5):
    # # mcdf = monte_carlo(
    # #     data=df,
    # #     interval=i,
    # #     n_periods=n_periods,
    # #     n_runs=1_000,
    # #     induce_volatility=True,
    # #     data_column_mappers={'dt': 'date'},
    # #     show_plot=True,
    # #     show_historical=True,
    # #     keep_historical=False
    # # )



    # # st = Combo(and_list=[TRIMA, Bollinger], or_list=[FibonacciRSI], data=df, column_mappers={'dt': 'date'})
    # # st = Combo(
    # #     or_list=[
    # #         ATR,
    # #         Bollinger,
    # #         DEMA,
    # #         DX,
    # #         EMA,
    # #         FibonacciRSI,
    # #         KAMA,
    # #         MACD,
    # #         MAMA,
    # #         Momentum,
    # #         RSI,
    # #         SMA,
    # #         StochasticRSI,
    # #         TRIMA
    # #     ], 
    # #     data=df, 
    # #     column_mappers={'dt': 'date'}
    # # )
    # st = KAMA(data=df, column_mappers={'dt': 'date'})
    # # st = KAMA(data=df, column_mappers={'dt': 'date'})
    
    # # st = Combo(or_list=[TRIMA, FibonacciRSI], data=mcdf)
    # # st = FibonacciRSI(data=mcdf)
    # # st = FibonacciRSI(df, column_mappers={'dt': 'date'})
    # # st = Bollinger(df, column_mappers={'dt': 'date'})
    # # st = StochasticRSI(data=mcdf)
    # # st = MACD(data=mcdf)

    # # fv = st.final_value(
    # #     cash=10_000,
    # #     commission=0.046,
    # #     # stop_loss=2,
    # #     # take_profit=10
    # # )
    # # print(f'{fv=}')


    # # result = st.backtest_optimize(        
    # #     cash=10_000,
    # #     commission=0.046,
    # #     # stop_loss=2,
    # #     # take_profit=10,
    # #     close_end_position=False,
    # #     top_n=3
    # # )
    # # for obt, combo in result:
    # #     print(f'\n\nBEST BACKTEST\n{combo=}')
    # #     print(obt)



    # buys = st.data[st.data[st.buy_column].notna()]
    # sells = st.data[st.data[st.sell_column].notna()]
    # print('\nBUY SIGNALS')
    # print(buys)
    # print('\nSELL SIGNALS')
    # print(sells)
    # print()
    # bt = st.backtest(
    #     cash=20_000,
    #     commission=100 - 99.54,
    #     # stop_loss=2,
    #     # take_profit=10,
    #     close_end_position=False
    # )
    # print(bt)
    # print()

    # # print(bt.trades_data)
    # # print()
    # # print(bt.trades_data.loc[bt.trades_data['pnl'] > 0, ['cash', 'shares', 'buy_price', 'sell_price', 'commission', 'pnl', 'duration']])
    