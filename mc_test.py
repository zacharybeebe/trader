


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

    # t = 'BTC'
    # i = '5m'

    # e = Engine(database_type='sqlite', engine_echo=False)#, drop_all_tables_on_init=True, create_all_tables_on_init=True)
    # trx = e.start_transaction()
    
    # with TimerInline('get ticker'):
    #     df = e.data(ticker=t, interval=i, start=datetime(2021, 1, 1), transaction=trx)
    #     print(df[:5].to_string())


    # trx.close()    
    # e.close()

    # strat = StrategyManager(data=df)

    # n_periods = 1000
    # with TimerInline('monte carlo'):
    #     mcdf = strat.monte_carlo(
    #         interval=i,
    #         n_periods=n_periods,
    #         n_runs=100,
    #         inplace=False
    #     )
    #     mcdf = mcdf[-n_periods - 5:].copy(deep=True)

    # s3 = StrategyManager.RSI.run_detached(
    #     data=mcdf,
    #     cash=35_000
    # )
    # # s3 = StrategyManager.SMA.run_detached(
    # #     data=mcdf,
    # #     cash=35_000
    # # )

    # sdf, info = s3.apply_trades()

    # print(sdf.tail())
    # debug_print_dict(info)
    # print()

    # sxx = StrategyManager(data=mcdf)

    # bt = sxx.backtest(
    #     strategy=getattr(StrategyManager, s3.__class__.__name__),
    #     cash=500_000
    # )
    # stats = bt.run()
    # print(stats)

    "tickers = [DOGE, BTC, ETH, LTC]"

    t = 'DOGE'
    i = '5m'


    # n_periods = n_periods_from_timedelta(td=timedelta(days=7), interval=i)
    n_periods = n_periods_from_timedelta(td=timedelta(weeks=2), interval=i)
    print(f'{n_periods=}')
    # n_periods_behind = n_periods_from_timedelta(td=timedelta(days=14), interval=i)
    # n_periods_behind = n_periods_from_timedelta(td=timedelta(weeks=2), interval=i)  # 5
    n_periods_behind = 6527
    print(f'{n_periods_behind=}')

    # TODO: Below is a patch knowing the interval, but make a dynamic way to find the satrt date based off of number of periods and an 'X' multipler of look-behind periods
    ss_start = datetime.now(tz=timezone.utc) - timedelta(minutes=5 * n_periods_behind)
    print(f'{ss_start=}')

    e = Engine(database_type='sqlite', engine_echo=False)#, drop_all_tables_on_init=True, create_all_tables_on_init=True)
    trx = e.start_transaction()
    
    with TimerInline('get ticker'):
        # df = e.data(ticker=t, interval=i, start=ss_start, transaction=trx)
        df = e.data(ticker=t, interval=i, start=None, transaction=trx)
        actual_start = df['dt'].iloc[0]
        print(df[-5:].to_string())
        print(f'{actual_start=}')


    trx.close()    
    e.close()

    strat = StrategyManager(data=df)
    # # strat_class = StrategyManager.SmaRsiAnd


    # # periods_behind = 20
    # # if hasattr(strat_class, 'trima_period'):
    # #     periods_behind = getattr(strat_class, 'trima_period')
    # # elif hasattr(strat_class, 'long_period'):
    # #     periods_behind = getattr(strat_class, 'long_period')
    # # elif hasattr(strat_class, 'short_period'):
    # #     periods_behind = getattr(strat_class, 'short_period')
    # # elif hasattr(strat_class, 'period'):
    # #     periods_behind = getattr(strat_class, 'period')
    # # print(f'{strat_class.__name__} periods_behind: {periods_behind}')

    # for mc_type in T.Strategy.MONTE_CARLO.__args__:

    # with TimerInline(f'Monte Carlo: "{mc_type}"'):

    for _ in range(4):
        mcdf = strat.monte_carlo_cls(
            dataframe=df,
            interval=i,
            n_periods=n_periods,
            n_runs=1_000,
            mc_type='normal',
            show_simulations=True,
            show_historical=True,
            induce_volatility=True
        )
        print(mcdf[['Close']][-5:].to_string())

    # with TimerInline('monte_normal_multi_cls'):
    #     mcdf = strat.monte_normal_multi_cls(
    #         dataframe=df,
    #         interval=i,
    #         n_periods=n_periods,
    #         n_runs=10_000
    #     )
    #     print(mcdf[['Close']][-5:].to_string())

    # with TimerInline('monte_normal_every_cls'):
    #     mcdf = strat.monte_normal_every_cls(
    #         dataframe=df,
    #         interval=i,
    #         n_periods=n_periods,
    #         n_runs=10_000
    #     )
    #     print(mcdf[['Close']][-5:].to_string())
    
    # with TimerInline('mc_choice_cls'):
    #     mcdf = strat.mc_pct_choice_cls(
    #         dataframe=df,
    #         interval=i,
    #         n_periods=n_periods,
    #         n_runs=10_000
    #     )
    #     print(mcdf[['Close']][-5:].to_string())
    
    # with TimerInline('mc_brownian_geo_cls'):
    #     mcdf = strat.mc_brownian_geo_cls(
    #         dataframe=df,
    #         interval=i,
    #         n_periods=n_periods,
    #         n_runs=10_000
    #     )
    #     print(mcdf[['Close']][-5:].to_string())
    
    # with TimerInline('monte carlo static 2'):
    #     mcdf2 = strat.monte_carlo_static_2(
    #         dataframe=df,
    #         interval=i,
    #         n_periods=n_periods,
    #         n_runs=10_000
    #     )
    #     print(mcdf2[['Close']][-5:].to_string())




    
    # TODO: Figure out how to find the best look-behind period for predicting future prices per crypto

    # s3 = strat_class.run_detached(
    #     data=mcdf,
    #     cash=35_000
    # )
    # # s3 = StrategyManager.SMA.run_detached(
    # #     data=mcdf,
    # #     cash=35_000
    # # )

    # sdf, info = s3.apply_trades()

    # print(sdf.tail())
    # debug_print_dict(info)
    # print()

    # sdf_copy = sdf.copy(deep=True)
    # sdf_copy.reset_index(inplace=True)
    # sdf_copy['LocalDate'] = sdf_copy['Date'].dt.tz_convert('America/Los_Angeles')
    # sdf_copy['LocalDate'] = sdf_copy['LocalDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # sdf_copy['Date'] = sdf_copy['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # fp = unique_file_name(f'{t}_{i}_{s3.__class__.__name__}', 'xlsx')
    # sdf_copy.to_excel(fp, index=False)

    # # sxx = StrategyManager(data=mcdf)

    # # bt = sxx.backtest(
    # #     strategy=getattr(StrategyManager, s3.__class__.__name__),
    # #     cash=500_000
    # # )
    # # stats = bt.run()
    # # print(stats)

    # s3 = strat_class.run_detached(
    #     data=df,
    #     cash=35_000
    # )
    # print()
    # print(s3.df[s3.display_cols].tail())

    # decision_now = s3.decide_now(holding=True)
    # print(f'{decision_now=}')

    # decision_predict = s3.decide_predict(interval=i, holding=True, next_close=98_627)
    # print(f'{decision_predict=}')
    