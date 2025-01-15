import pandas as pd
import talib
from datetime import datetime, timedelta
from strategy_manager import StrategyManager
from strategy_manager_old import StrategyManager as Sld
from numbers import Number

from .utils.utils import nan

airlines = {
    'AAL': ['American Airlines', {
        'sma': {
            'short_window': {
                7: [84, 85, 86],
                8: [81, 82, 83, 84, 85, 86],
                9: [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
                10: [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
                11: [77, 78, 79, 80, 81, 82, 83],
                12: [76, 77, 78]
            }
        }
    }],
    'ALK': 'Alaska Airlines',
    'DAL': 'Delta Airlines',
    'LUV': 'Southwest Airlines',
    'UAL': 'United Airlines'
}

if __name__ == '__main__':

    # s = pd.Series([0, 0, 0, 0, 1])
    # x = 0
    #
    # series1 = (
    #     s.values if isinstance(s, pd.Series) else
    #     (s, s) if isinstance(s, Number) else
    #     s)
    # series2 = (
    #     x.values if isinstance(x, pd.Series) else
    #     (x, x) if isinstance(x, Number) else
    #     x)
    # try:
    #     print('Going Ahead')
    #     print(series1[-1] > series2[-1])
    #     print(series1[-2] < series2[-2])
    #     print(series1[-2] < series2[-2] and series1[-1] > series2[-1])
    # except IndexError:
    #     print('Index Error')
    #
    # print()

    strat = StrategyManager(
        ticker='ALK',
        end_date=datetime.now(),
        start_days_prior=729,   #729
        interval='1h'
    )

    # strat = StrategyManager(
    #     ticker='ALK',
    #     end_date=datetime.now(),
    #     start_days_prior=59,   #729
    #     interval='30m'
    # )

    # data = strat.get_data()

    # for method_name in StrategyManager.__dict__:
    #     if method_name.startswith('cdl'):
    #         print(f'{method_name=}')
    #         method = getattr(StrategyManager, method_name)
    #         data = method(data=data)
    #         print(data.loc[data[method_name] != 0].to_string())
    #         print()

    #
    # data = strat.cdl_inverted_hammer(data=data)
    # print(data.loc[data['cdl_inverted_hammer'] == 1].to_string())
    #
    # data = strat.cdl_hammer(data=data)
    # print(data.loc[data['cdl_hammer'] == 1].to_string())
    #
    # data = strat.cdl_piercing(data=data)
    # print(data.loc[data['cdl_piercing'] == 1].to_string())
    #
    # data = strat.cdl_morning_doji_star(data=data)
    # print(data.loc[data['cdl_morning_doji_star'] == 1].to_string())
    #
    # data = strat.cdl_morning_star(data=data)
    # print(data.loc[data['cdl_morning_star'] == 1].to_string())

    # sld = Sld(ticker=strat.ticker, start_date=strat.start_date, end_date=strat.end_date, interval=strat.interval)
    #
    # #data = strat.get_data()
    #
    # data = sld.macd(short_window=20, long_window=25, signal_window=9)
    # data.set_index('date', drop=True, inplace=True)
    #
    # data['short_ema'] = talib.EMA(data['adj_close'], 20)
    # data['long_ema'] = talib.EMA(data['adj_close'], 25)
    # data['macd'], data['macd_s'], data['macd_h'] = talib.MACD(data['adj_close'], 20, 25, 9)
    #
    #
    # print(data.to_string())

    # combo = strat.strategy_combo(
    #     [strat.Bollinger, {'period': 25, 'stddev_up': 1.75, 'stddev_down': 1.75}],
    #     [strat.DX, {'period': 14, 'adx_threshold': 25}],
    #     [strat.EMA, {'short_period': 10, 'long_period': 30}],
    #     [strat.MACD, {'short_period': 20, 'long_period': 25, 'signal_period': 9}],
    #     [strat.Momentum, {'period': 15, 'buy_threshold': 1.08, 'sell_threshold': 0.92}],
    #     [strat.SMA, {'short_period': 20, 'long_period': 25}],
    #     [strat.RSI, {'period': 15, 'overbought': 65, 'oversold': 25}],
    #     buy_sell_at_strength=4
    # )

    # combo = strat.strategy_combo(
    #     [strat.Bollinger, None],
    #     [strat.DX, None],
    #     [strat.EMA, None],
    #     [strat.MACD, None],
    #     [strat.Momentum, None],
    #     [strat.RSI, None],
    #     [strat.SMA, None],
    #     [strat.StochasticRSI, None],
    #     buy_sell_at_strength=3
    # )

    # candles = strat.strategy_candles(
    #     strat.cdl_inverted_hammer,
    #     strat.cdl_morning_doji_star,
    #     strat.cdl_3_black_crows,
    #     strat.cdl_3_white_soldiers,
    #     strat.cdl_dark_cloud_cover,
    #     strat.cdl_evening_star,
    #     class_name='Candles',
    # )
    #
    # candle_combo = strat.strategy_combo(
    #     [candles, None],
    #     [strat.SMA, {'short_period': 20, 'long_period': 25}],
    #     buy_sell_at_strength=2
    # )

    bt = strat.backtest(strategy=strat.NewLow)
    stats = bt.run()

    # bt = strat.backtest(strategy=strat.TRIMA)
    # stats = bt.run(period=98)

    # stats = bt.optimize(
    #     period=range(2, 201, 2),
    #     skip_sell=False,
    #     maximize='Return [%]'
    # )

    # stats = bt.optimize(
    #     short_period=range(50, 71, 2),
    #     long_period=range(50, 101, 2),
    #     trima_period=range(90, 111, 2),
    #     skip_sell=True,
    #     maximize='Return [%]',
    #     constraint=lambda param: param.short_period < param.long_period
    # )

    # stats, heatmap = bt.optimize(
    #     sell_threshold=[i / 100 for i in range(85, 99, 1)],
    #     buy_threshold=[i / 100 for i in range(101, 121, 1)],
    #     maximize='Return (Ann.) [%]',
    #     # constraint=lambda param: param.short_period < param.long_period,
    #     return_heatmap=True
    # )


    # Previous 927 - 945
    # -- 1068 - 1150

    # 932 - 960
    # 1030 - 1085
    sell = list(range(927, 945, 1))
    buy = list(range(1068, 1150, 1))
    combos = len(sell) * len(buy)
    print(f'{combos=}')

    stats, heatmap = bt.optimize(
        sell_threshold=[i / 1000 for i in sell],
        buy_threshold=[i / 1000 for i in buy],
        maximize='Return (Ann.) [%]',
        # constraint=lambda param: param.short_period < param.long_period,
        return_heatmap=True
    )

    print(stats)
    print()
    print(stats._strategy)
    print('\n\n')


    print(heatmap)
    strat.show_heatmap(heatmap_from_backtest=heatmap, y_axis_param='buy_threshold', x_axis_param='sell_threshold')

    bt.plot()

