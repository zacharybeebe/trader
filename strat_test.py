


if __name__ == '__main__':
    
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

    from trader.backend.trading.strategy_manager import StrategyManager

    "tickers = [DOGE, BTC, ETH, LTC]"

    t = 'DOGE'
    i = '5m'

    # # n_periods_behind = 6527
    n_periods_behind = n_periods_from_timedelta(td=timedelta(weeks=5), interval=i)  # 5
    print(f'{n_periods_behind=}')

    # # TODO: Below is a patch knowing the interval, but make a dynamic way to find the satrt date based off of number of periods and an 'X' multipler of look-behind periods
    start = datetime.now(tz=timezone.utc) - timedelta(minutes=5 * n_periods_behind)
    # start = None
    print(f'{start=}')

    e = Engine(database_type='sqlite', engine_echo=False)#, drop_all_tables_on_init=True, create_all_tables_on_init=True)
    trx = e.start_transaction()
    
    with TimerInline('get ticker'):
        df = e.data(ticker=t, interval=i, start=start, transaction=trx)
        actual_start = df['dt'].iloc[0]
        print(df[-5:].to_string())
        print(f'{actual_start=}')
    
    print(f'\nstart_price = ${df["close"].iloc[0]:,.6f} @ {df["dt"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'end_price = ${df["close"].iloc[-1]:,.6f} @ {df["dt"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")}\n')


    trx.close()    
    e.close()

    s_class = StrategyManager.FibonacciRSI
    # s_class = StrategyManager.SMA

    strat = StrategyManager(
        data=df, 
        share_fraction=1
        # share_fraction=1_000_000
    )
    print(strat.data)

    bt = strat.backtest(
        strategy=s_class,
        # strategy=strat.SMA,
        cash=10_000
    )
    stats = bt.run()#buy_threshold=45)
    strat.normalize_data(inplace=True)
    print(stats)
    print()
    # print(strat.data)
    # was_fract_trades = strat.trades_from_stats(stats, was_run_as_fraction=True)
    # print(was_fract_trades.to_string())
    bt.plot()


    s_instance = s_class.run_detached(
        data=df,
        cash=10_000
    )

    data, info, trades_df = s_instance.apply_trades(end_sell=True, commission_pct=0.46, plot=True)
    debug_print_dict(info)
    