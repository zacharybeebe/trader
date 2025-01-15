

if __name__ == '__main__':
    import numpy as np
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
        Optional,
        T
    )

    from trader.backend.trading.strategy_manager import StrategyManager



    def best_look_behinds(
            dataframe: pd.DataFrame, 
            top_n: int = 5,
            look_ahead_periods: int = 1,
            interval: T.Trade.INTERVAL = '5m', 
            n_runs: int = 1_000, 
            trend_threshold: float = 0.025,
            max_look_behind: Optional[int] = None,
            mc_type: T.Strategy.MONTE_CARLO = 'normal'
    ) -> dict:
        len_df = len(dataframe)
        last_idx = len_df - 1
        last_price = dataframe['close'].iloc[last_idx]
        print(f'last_price=${last_price:,.6f}')

        if look_ahead_periods <= 2:
            look_behind_idx = len_df - look_ahead_periods - 3
        else:
            look_behind_idx = len_df - look_ahead_periods - 1
        
        if max_look_behind is None:
            max_look_behind = 0
        else:
            max_look_behind = look_behind_idx - max_look_behind
            if max_look_behind > len_df - 1:
                max_look_behind = 0

        best_guesses_look_behind = []

        with TimerInline('Running Best Look Behinds') as ti:
            while look_behind_idx > max_look_behind:
                if look_behind_idx % 1_000 == 0:
                    ti.subtime(label='This 1000')
                    print(f'{look_behind_idx} / {max_look_behind}')

                this_run_data = dataframe[look_behind_idx:len_df - 1]
                trend = StrategyManager.trend(
                    start_price=this_run_data['close'].iloc[0],
                    end_price=this_run_data['close'].iloc[-1],
                    trend_threshold=trend_threshold
                )
                mcdf = StrategyManager.monte_carlo_cls(
                    dataframe=this_run_data,
                    interval=interval,
                    n_periods=look_ahead_periods,
                    n_runs=n_runs,
                    mc_type=mc_type,
                    show_simulations=False,
                    show_historical=False
                )
                predicted_price = mcdf['Close'].iloc[-1]
                if len(best_guesses_look_behind) < top_n:
                    best_guesses_look_behind.append({
                        'look_behind': len_df - look_behind_idx - 1,
                        'price': predicted_price,
                        'trend': trend
                    })
                else:
                    replace_idx = None
                    predicted_diff = abs(predicted_price - last_price)
                    for idx in range(len(best_guesses_look_behind)):
                        if predicted_diff < abs(best_guesses_look_behind[idx]['price'] - last_price):
                            replace_idx = idx
                            break
                    if replace_idx is not None:
                        best_guesses_look_behind[replace_idx] = {
                            'look_behind': len_df - look_behind_idx - 1,
                            'price': predicted_price,
                            'trend': trend
                        }      
                look_behind_idx -= 1
        
        best_guesses_look_behind = sorted(best_guesses_look_behind, key=lambda x: x['price'])
        return best_guesses_look_behind



    "tickers = [DOGE, BTC, ETH, LTC]"

    t = 'DOGE'
    i = '5m'

    look_ahead_periods = n_periods_from_timedelta(td=timedelta(days=1), interval=i)
    # look_ahead_periods = 1
    print(f'{look_ahead_periods=}')

    # e = Engine(database_type='sqlite', engine_echo=False)#, drop_all_tables_on_init=True, create_all_tables_on_init=True)
    # trx = e.start_transaction()
    
    # with TimerInline('get ticker'):
    #     df = e.data(ticker=t, interval=i, start=None, transaction=trx)
    #     print(df[['dt', 'close']])
    #     print()
    # trx.close()    
    # e.close()

    # best_guesses_look_behind = best_look_behinds(
    #     dataframe=df,
    #     look_ahead_periods=look_ahead_periods,
    #     interval=i,
    #     n_runs=1_000,
    #     trend_threshold=0.025,
    #     # max_look_behind=10_000,
    #     max_look_behind=None,
    #     mc_type='normal'
    # )
    # for bg in best_guesses_look_behind:
    #     print(f'{bg=}')