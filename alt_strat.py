


if __name__ == '__main__':

    s = None

    print(f'{s is None=}')
    print(f'{s is not None=}')
    print(f'{s == None=}')
    print(f'{s != None=}')

    # import talib
    # import numpy as np
    # from scipy.signal import find_peaks
    # import matplotlib.pyplot as plt
    # from trader.backend.data.engine import (
    #     Engine, 
    #     pd, 
    #     isna, 
    #     datetime, 
    #     datetime_parse, 
    #     get_dt_at_n_periods,
    #     n_periods_from_timedelta,
    #     timezone, 
    #     TimerInline, 
    #     timedelta, 
    #     generate_uuid, 
    #     yf, 
    #     utc_to_pst, 
    #     debug_print_dict, 
    #     unique_file_name,
    #     T
    # )

    # from trader.backend.trading.strategy_manager import StrategyManager

    # "tickers = [DOGE, BTC, ETH, LTC]"

    # t = 'BTC'
    # i = '5m'

    # # # n_periods_behind = 6527
    # n_periods_behind = n_periods_from_timedelta(td=timedelta(weeks=5), interval=i)  # 5
    # print(f'{n_periods_behind=}')

    # # # TODO: Below is a patch knowing the interval, but make a dynamic way to find the satrt date based off of number of periods and an 'X' multipler of look-behind periods
    # start = datetime.now(tz=timezone.utc) - timedelta(minutes=5 * n_periods_behind)
    # # start = None
    # print(f'{start=}')

    # e = Engine(database_type='sqlite', engine_echo=False)#, drop_all_tables_on_init=True, create_all_tables_on_init=True)
    # trx = e.start_transaction()
    
    # with TimerInline('get ticker'):
    #     df = e.data(ticker=t, interval=i, start=start, transaction=trx)
    #     actual_start = df['dt'].iloc[0]
    #     print(df[-5:].to_string())
    #     print(f'{actual_start=}')
    
    # print(f'\nstart_price = ${df["close"].iloc[0]:,.6f} @ {df["dt"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")}')
    # print(f'end_price = ${df["close"].iloc[-1]:,.6f} @ {df["dt"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")}\n')


    # trx.close()    
    # e.close()

    # df['sma_400'] = talib.SMA(df['close'], 400)
    # df['sma_is_peak'] = np.nan
    # df['sma_is_valley'] = np.nan
    # print(df[['dt', 'close', 'sma_400']])

    # # print(f'{len(df)=}')
    # # peaks, props = find_peaks(df['close'], prominence=1)
    # # print(f'{peaks=}')
    # # print(f'{len(peaks)=}')
    # # debug_print_dict(props)

    # last_peak = None
    # last_valley = None
    # same_threshold = 0.01
    # different_threshold = 0.02
    # different_for_peak_used = False
    # different_for_valley_used = False
    # last_either_type = None
    # same_threshold_for_multiple = 0.0525

    # for idx in df.index:
    #     if 0 < idx < len(df) - 1:
    #         prev_neighbor = df['sma_400'].iloc[idx - 1]
    #         current = df['sma_400'].iloc[idx]
    #         next_neighbor = df['sma_400'].iloc[idx + 1]
    #         is_peak = False
    #         is_valley = False
    #         # Check if the current value is a peak (is it larger than its neighbors)
    #         # Check as well if the last found type was a valley (multiple peaks in a row is not allowed)
    #         if prev_neighbor < current > next_neighbor:
    #             # Set right away if there is no last valley
    #             if last_peak is None:
    #                 is_peak = True
                
    #             else:
    #                 # Check that the last found type was a valley but also compare it the the same_threshold_for_multiple
    #                 # usually multiple peaks in a row are not allowed, but if the peak is higher than the last peak by the
    #                 # same_threshold_for_multiple, then it is allowed
    #                 if last_either_type == 'valley':
    #                     # Check if the peak has surpassed the same_threshold from the last peak
    #                     if current > last_peak * (1 + same_threshold):
    #                         is_peak = True     

    #                     # Check if the peak has surpassed the different_threshold from the last valley
    #                     # different_for_peak_used is used to prevent the peak from being set multiple times
    #                     # different_for_peak_used will reset when a new valley is found
    #                     elif last_valley is not None and current > last_valley * (1 + different_threshold) and not different_for_peak_used:
    #                         is_peak = True        
    #                         different_for_peak_used = True
                    
    #                 else:
    #                      if current > last_peak * (1 + same_threshold_for_multiple):
    #                         is_peak = True


    #         # Check if the current value is a valley (is it smaller than its neighbors)
    #         # Check as well if the last found type was a peak (multiple valleys in a row is not allowed)
    #         elif prev_neighbor > current < next_neighbor:
    #             # Set right away if there is no last valley
    #             if last_valley is None:
    #                 is_valley = True

    #             else:
    #                 # Check that the last found type was a peak but also compare it the the same_threshold_for_multiple
    #                 # usually multiple valleys in a row are not allowed, but if the valley is lower than the last valley by the
    #                 # same_threshold_for_multiple, then it is allowed
    #                 if last_either_type == 'peak':
    #                     # Check if the valley has surpassed the same_threshold from the last valley
    #                     if current < last_valley * (1 - same_threshold):
    #                         is_valley = True
                        
    #                     # Check if the valley has surpassed the different_threshold from the last peak
    #                     # different_for_valley_used is used to prevent the valley from being set multiple times
    #                     # different_for_valley_used will reset when a new peak is found
    #                     elif last_peak is not None and current < last_peak * (1 - different_threshold) and not different_for_valley_used:
    #                         is_valley = True
    #                         different_for_valley_used = True
                    
    #                 else:
    #                      if current < last_valley * (1 - same_threshold_for_multiple):
    #                         print(f'{current=} {last_valley=} {last_valley * (1 - same_threshold_for_multiple)}')
    #                         is_valley = True

            
    #         if is_peak:
    #             df.at[idx, 'sma_is_peak'] = current
    #             last_peak = current
    #             different_for_valley_used = False
    #             last_either_type = 'peak'
            
    #         if is_valley:
    #             df.at[idx, 'sma_is_valley'] = current
    #             last_valley = current
    #             different_for_peak_used = False
    #             last_either_type = 'valley'

    

    # print(df.loc[df['sma_is_peak'].notna() ,['dt', 'close', 'sma_400', 'sma_is_peak', 'sma_is_valley']])
    # print()
    # print(df.loc[df['sma_is_valley'].notna() ,['dt', 'close', 'sma_400', 'sma_is_peak', 'sma_is_valley']])
    # print()

    # plt.figure(figsize=(15, 8))
    # plt.plot(df['dt'], df['close'], label='Close', color='black', alpha=0.5)
    # plt.plot(df['dt'], df['sma_400'], label='SMA 400', color='red', alpha=0.75)
    # plt.scatter(df['dt'], df['sma_is_peak'], color='green', marker='^', label='Peak')
    # plt.scatter(df['dt'], df['sma_is_valley'], color='red', marker='v', label='Valley')

    # plt.show()
    