



if __name__ == '__main__':
    from strategy_manager import StrategyManager
    from monte_carlo import MonteCarloPercentChange
    from datetime import datetime
    from statistics import mean, stdev
    import pandas as pd
    from typing import Literal

    import pandas_market_calendars as mcal

    def convert_pd_timestamp_to_datetime_unaware(timestamp: pd.Timestamp) -> datetime:
        converted_dt = timestamp.to_pydatetime()
        return datetime(
            year=converted_dt.year,
            month=converted_dt.month,
            day=converted_dt.day,
            hour=converted_dt.hour,
            minute=converted_dt.minute,
            second=converted_dt.second,
            microsecond=converted_dt.microsecond,
        )

    def get_next_market_datetime_by_interval(
            current_datetime: datetime,
            interval: Literal['1m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    ) -> datetime:
        def check_addition(base_kwargs: dict, addition: int) -> None:
            if interval.endswith('m'):
                if base_kwargs['minute'] + addition > 60:


        def update_base_kwargs(dt: datetime, keyword: str, addition: int):
            base_kwargs = dict(
                year=dt.year,
                month=dt.month,
                day=dt.day,
                hour=dt.hour,
                minute=dt.minute,
                second=dt.second,
                microsecond=dt.microsecond
            )
            base_kwargs[keyword] += addition
            return base_kwargs

        if interval == '1m':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='minute', addition=1)
        elif interval == '5m':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='minute', addition=5)
        elif interval == '15m':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='minute', addition=15)
        elif interval == '30m':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='minute', addition=30)
        elif interval == '60m':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='minute', addition=60)
        elif interval == '90m':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='minute', addition=90)
        elif interval == '1h':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='hour', addition=1)
        elif interval == '1d':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='day', addition=1)
        elif interval == '5d':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='day', addition=5)
        elif interval == '1wk':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='day', addition=7)
        elif interval == '1mo':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='month', addition=1)
        else:  # interval == '3mo':
            check_date_lambda = lambda dt: update_base_kwargs(dt=dt, keyword='month', addition=3)

        new_dt = current_datetime
        new_dt = datetime(**check_date_lambda(dt=new_dt))

        nyse = mcal.get_calendar('NYSE')
        results = nyse.schedule(start_date=new_dt, end_date=new_dt, tz=nyse.tz.zone)
        while results.empty:
            new_dt = datetime(**check_date_lambda(dt=new_dt))
            results = nyse.schedule(start_date=new_dt, end_date=new_dt, tz=nyse.tz.zone)

        market_open = convert_pd_timestamp_to_datetime_unaware(timestamp=results['market_open'].iloc[0])
        market_close = convert_pd_timestamp_to_datetime_unaware(timestamp=results['market_close'].iloc[0])

        while not (market_open <= new_dt <= market_close):
            new_dt = datetime(**check_date_lambda(dt=new_dt))
            results = nyse.schedule(start_date=new_dt, end_date=new_dt, tz=nyse.tz.zone)
            market_open = convert_pd_timestamp_to_datetime_unaware(timestamp=results['market_open'].iloc[0])
            market_close = convert_pd_timestamp_to_datetime_unaware(timestamp=results['market_close'].iloc[0])
        return new_dt


    # dt = datetime(year=2024, month=1, day=1)
    nyse = mcal.get_calendar('NYSE')
    # dt = datetime.now(tz=nyse.tz)
    dt = datetime(year=2024, month=6, day=14, hour=15, minute=30, second=0, microsecond=0)
    print(f'{dt=}')
    results = nyse.schedule(start_date=dt, end_date=dt, tz=nyse.tz.zone)
    market_open = convert_pd_timestamp_to_datetime_unaware(timestamp=results['market_open'].iloc[0])
    market_close = convert_pd_timestamp_to_datetime_unaware(timestamp=results['market_close'].iloc[0])
    print(results)
    print(f'{market_open=}')
    print(f'{market_close=}')
    print(dt <= market_close)

    strat = StrategyManager(
        ticker='ALK',
        end_date=datetime.now(),
        start_days_prior=729,  # 729
        interval='1h'
    )
    data = strat.get_data()
    print(data[-5:].to_string())
    dttt = convert_pd_timestamp_to_datetime_unaware(timestamp=data.index[-1])

    print(dttt)
    print(type(dttt))
    print(dttt <= market_close)
    print(strat.interval)
    next_dt = get_next_market_datetime_by_interval(current_datetime=dttt, interval=strat.interval)
    print(next_dt)
    print(type(next_dt))



    #
    # monte = MonteCarloPercentChange(data=data, price_column='Adj Close')
    # mc_df = monte.run(n_periods_forward=1_000, np_distribution='normal')
    #
    # print(mc_df[:20].to_string())

    # ty_data = {
    #     'normal': {
    #         'all_change': [],
    #         'all_first': [],
    #         'all_last': [],
    #         'mean_last': None,
    #         'stdev_last': None,
    #         'mean_change': None,
    #         'stdev_change': None
    #     },
    #     'logistic': {
    #         'all_change': [],
    #         'all_first': [],
    #         'all_last': [],
    #         'mean_last': None,
    #         'stdev_last': None,
    #         'mean_change': None,
    #         'stdev_change': None
    #     },
    #     'laplace': {
    #         'all_change': [],
    #         'all_first': [],
    #         'all_last': [],
    #         'mean_last': None,
    #         'stdev_last': None,
    #         'mean_change': None,
    #         'stdev_change': None
    #     }
    # }
    #
    # for gg in ty_data:
    #     for i in range(10):
    #         monte = MonteCarloPercentChange(data=data, price_column='Adj Close')
    #         mc_df = monte.run(n_periods_forward=1_000, np_distribution=gg)
    #         # loc, scale, size methods are
    #         # normal
    #         # logistic
    #         # laplace
    #         first = mc_df['predict_adj_close'].iloc[0]
    #         last = mc_df['predict_adj_close'].iloc[-1]
    #         overall_pct = ((last - first) / last) * 100
    #         ty_data[gg]['all_change'].append(overall_pct)
    #         ty_data[gg]['all_first'].append(first)
    #         ty_data[gg]['all_last'].append(last)
    #     ty_data[gg]['mean_last'] = mean(ty_data[gg]['all_last'])
    #     ty_data[gg]['stdev_last'] = stdev(ty_data[gg]['all_last'])
    #     ty_data[gg]['mean_change'] = mean(ty_data[gg]['all_change'])
    #     ty_data[gg]['stdev_change'] = stdev(ty_data[gg]['all_change'])
    #
    # for s in ty_data:
    #     print(s)
    #     for k, v in ty_data[s].items():
    #         print(f'{k:20}{v}')



