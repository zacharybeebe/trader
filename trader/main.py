import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import yfinance as yf
#
# from datetime import datetime, date, timedelta
#
# import math


if __name__ == '__main__':



    # # alaska = yf.Ticker(ticker='ALK')
    #
    # alk_df = pd.read_csv('ALK_30_days_0410_0509.csv')
    #
    # all_closes = alk_df['Close'].to_list()
    #
    # filename = 'aggregate_w_exp_mod_and_mul_mod.txt'
    #
    # # Is in cents gained or lost
    # #buy_pattern = 300
    # #sell_pattern = -600
    # #minutes_back_to_look = 7
    #
    # buy_sell_cap = 9999
    # best_pattern = None
    # best_minute = None
    # best_end_money = None
    # ending_inputs = None
    #
    # #sell_multipliers = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    #
    # line = 1
    # count = 1
    # max_dot = 200
    #
    # #for sell_multiplier in sell_multipliers:
    #
    # #buy_sell_patterns = [[i, -i*sell_multiplier] for i in range(100, 551)]
    # buy_sell_patterns = [[i, -i * 2] for i in range(100, 551)]
    # lookbacks = range(4, 31)
    # exponent_mods = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
    #
    # iters = len(buy_sell_patterns) * len(lookbacks) * len(exponent_mods)# * len(sell_multipliers)
    # print(f'{line * max_dot} / {iters:,.0f} ', sep='', end='')
    #
    # for buy_pattern, sell_pattern in buy_sell_patterns:
    #     for minutes_back_to_look in lookbacks:
    #         for exponent_mod in exponent_mods:
    #             if count == 200:
    #                 count = 1
    #                 line += 1
    #                 print(f'\n{line * max_dot} / {iters:,.0f} ', sep='', end='')
    #             else:
    #                 print('.', sep='', end='')
    #                 count += 1
    #
    #             # values = {
    #             #     'Date': [],
    #             #     'Buys': [],
    #             #     'Sells': [],
    #             #     'Shares': [],
    #             #     'Invest': []
    #             # }
    #
    #             initial = True
    #             previous_close = None
    #             current_close = None
    #             closes = []
    #             buys = 0
    #             sells = 0
    #
    #             start_investment_base = 2_400
    #             start_investment = start_investment_base
    #             shares = 0
    #             price_adjust = .99865
    #
    #             for zdx, close_i in enumerate(all_closes):
    #                 buy = False
    #                 sell = False
    #
    #                 if zdx == 0 and initial:
    #                     initial = False
    #                     previous_close = close_i
    #                     closes.append(previous_close)
    #                     continue
    #
    #                 current_close = close_i
    #                 closes.append(current_close)
    #
    #                 if len(closes) > minutes_back_to_look:
    #                     aggregate = 0
    #                     up_bucket = 0
    #                     down_bucket = 0.0
    #
    #                     look_closes = closes[-minutes_back_to_look:]
    #
    #                     for cdx in range(1, len(look_closes)):
    #                         diff_cls = float(look_closes[cdx] - look_closes[cdx - 1])
    #                         if diff_cls < 0:
    #                             g = abs(diff_cls * 100) ** exponent_mod
    #                             down_bucket -= g
    #                             aggregate -= g
    #                         elif diff_cls > 0:
    #                             g = abs(diff_cls * 100) ** exponent_mod
    #                             up_bucket += g
    #                             aggregate += g
    #
    #                     if aggregate >= buy_pattern and buys < buy_sell_cap:
    #                         buy = True
    #                     elif aggregate <= sell_pattern and sells < buy_sell_cap:
    #                         sell = True
    #
    #                     if buy and shares == 0:
    #                         shares = start_investment / (current_close / price_adjust)
    #                         start_investment = 0
    #                         buys += 1
    #
    #                     if sell and shares > 0:
    #                         start_investment = shares * (current_close * price_adjust)
    #                         shares = 0
    #                         sells += 1
    #
    #                     # if up_bucket >= buy_pattern and buys < buy_sell_cap:
    #                     #     buy = True
    #                     # elif down_bucket <= sell_pattern and sells < buy_sell_cap:
    #                     #     sell = True
    #                     #
    #                     # if buy and shares == 0:
    #                     #     shares = start_investment / (current_close / price_adjust)
    #                     #     start_investment = 0
    #                     #     buys += 1
    #                     #
    #                     # if sell and shares > 0:
    #                     #     start_investment = shares * (current_close * price_adjust)
    #                     #     shares = 0
    #                     #     sells += 1
    #
    #                 # values['Date'].append(alk_df['Date'][i].split(' ')[1])
    #                 # values['Buys'].append(buys)
    #                 # values['Sells'].append(sells)
    #                 # values['Shares'].append(round(shares, 2))
    #                 # values['Invest'].append(f'${start_investment:,.2f}')
    #
    #                 previous_close = current_close
    #
    #             #df = pd.DataFrame(data=values)
    #             # end_shares = df['Shares'][len(df) - 1]
    #             if shares > 0:
    #                 end_money = shares * previous_close
    #             else:
    #                 end_money = start_investment
    #             pct = ((end_money - start_investment_base) / start_investment_base) * 100
    #             annual = (1 + (pct / 100)) ** 12
    #             five_yr = (1 + (pct / 100)) ** 60
    #             ten_yr = (1 + (pct / 100)) ** 120
    #
    #             if best_pattern is None:
    #                 best_pattern = {'buy_pattern': buy_pattern, 'sell_pattern': sell_pattern}
    #                 best_minute = minutes_back_to_look
    #                 best_end_money = end_money
    #                 ending_inputs = [
    #                     f'Best Buy Pattern:\t\t\t\t\t\t\t\t\t\t\t{buy_pattern}',
    #                     f'Best Sell Pattern:\t\t\t\t\t\t\t\t\t\t\t{sell_pattern}',
    #                     f'Best Minutes to Look Back:\t\t\t\t\t\t\t\t\t{minutes_back_to_look}',
    #                     f'Best Exponent Modification:\t\t\t\t\t\t\t\t\t{exponent_mod}',
    #                     f'Best Ending Investment:\t\t\t\t\t\t\t\t\t\t${best_end_money:,.2f}',
    #                     f'Best Percent Change over 30 days of trading:\t\t\t\t\t{pct:.2f}%',
    #                     f'Best Annual Percent if maintained:\t\t\t\t\t\t\t{(annual - 1) * 100:.2f}%',
    #                     f'Best If maintain {pct:.2f}% per month, Invest after 1 Year:\t\t${start_investment_base * annual:,.2f}',
    #                     f'Best If maintain {pct:.2f}% per month, Invest after 5 Year:\t\t${start_investment_base * five_yr:,.2f}',
    #                     f'Best If maintain {pct:.2f}% per month, Invest after 10 Year:\t${start_investment_base * ten_yr:,.2f}\n'
    #                 ]
    #             elif end_money > best_end_money:
    #                 best_pattern = {'buy_pattern': buy_pattern, 'sell_pattern': sell_pattern}
    #                 best_minute = minutes_back_to_look
    #                 best_end_money = end_money
    #                 ending_inputs = [
    #                     f'Best Buy Pattern:\t\t\t\t\t\t\t\t\t\t{buy_pattern}',
    #                     f'Best Sell Pattern:\t\t\t\t\t\t\t\t\t\t{sell_pattern}',
    #                     f'Best Minutes to Look Back:\t\t\t\t\t\t\t\t{minutes_back_to_look}',
    #                     f'Best Exponent Modification:\t\t\t\t\t\t\t\t\t{exponent_mod}',
    #                     f'Best Ending Investment:\t\t\t\t\t\t\t\t\t${best_end_money:,.2f}',
    #                     f'Percent Change over last month of trading:\t\t\t{pct:.2f}%',
    #                     f'Annual Percent if maintained:\t\t\t\t\t\t\t{(annual - 1) * 100:.2f}%',
    #                     f'If maintain {pct:.2f}% per month, Invest after 1 Year:\t\t${start_investment_base * annual:,.2f}',
    #                     f'If maintain {pct:.2f}% per month, Invest after 5 Year:\t\t${start_investment_base * five_yr:,.2f}',
    #                     f'If maintain {pct:.2f}% per month, Invest after 10 Year:\t\t${start_investment_base * ten_yr:,.2f}\n'
    #                     ]
    #
    # with open(filename, 'w') as f:
    #     f.writelines('\n'.join(ending_inputs))
    #
    # print()
    # for i in ending_inputs:
    #     print(i)






    # alaska = yf.Ticker(ticker='ALK')
    #
    # target_low_start_hour_1 = [8, 30]
    # target_low_end_hour_1 = [9, 30]
    #
    # target_high_start_hour_1 = [11, 0]
    # target_high_end_hour_1 = [13, 0]
    #
    # # ---------------------------------------------------------------------------------------------------------------
    #
    # target_low_start_hour_2 = [6, 30]
    # target_low_end_hour_2 = [7, 20]
    #
    # target_high_start_hour_2 = [7, 50]
    # target_high_end_hour_2 = [8, 10]
    #
    # utc_adjust = 7
    #
    # # Adjusted for UTC
    # low_start_hour_1 = {'hour': target_low_start_hour_1[0] + utc_adjust, 'minute': target_low_start_hour_1[1]}
    # low_end_hour_1 = {'hour': target_low_end_hour_1[0] + utc_adjust, 'minute': target_low_end_hour_1[1]}
    #
    # high_start_hour_1 = {'hour': target_high_start_hour_1[0] + utc_adjust, 'minute': target_high_start_hour_1[1]}
    # high_end_hour_1 = {'hour': target_high_end_hour_1[0] + utc_adjust, 'minute': target_high_end_hour_1[1]}
    #
    # #----------------------------------------------------------------------------------------------------------------
    #
    # low_start_hour_2 = {'hour': target_low_start_hour_2[0] + utc_adjust, 'minute': target_low_start_hour_2[1]}
    # low_end_hour_2 = {'hour': target_low_end_hour_2[0] + utc_adjust, 'minute': target_low_end_hour_2[1]}
    #
    # high_start_hour_2 = {'hour': target_high_start_hour_2[0] + utc_adjust, 'minute': target_high_start_hour_2[1]}
    # high_end_hour_2 = {'hour': target_high_end_hour_2[0] + utc_adjust, 'minute': target_high_end_hour_2[1]}
    #
    # weekdays = [0, 1, 2, 3, 4]
    # days_prior = 30
    # today = datetime.now()
    # today_minus_days = today - timedelta(days=days_prior)
    #
    # start_investment = 2_400
    # price_adjust = .99865
    #
    # values = {
    #     'Date': [],
    #     'Low_1': [],
    #     'High_1': [],
    #     'Diff_1': [],
    #     'Pct_1': [],
    #     '_____': [],
    #     'Low_2': [],
    #     'High_2': [],
    #     'Diff_2': [],
    #     'Pct_2': [],
    #     '-----': [],
    #     'Invest': []
    # }
    #
    #
    # for _ in range(days_prior):
    #     if today_minus_days.weekday() in weekdays:
    #         yr = today_minus_days.year
    #         mo = today_minus_days.month
    #         dy = today_minus_days.day
    #
    #         low_start_1 = pd.Timestamp(year=yr, month=mo, day=dy, tz='UTC', **low_start_hour_1)
    #         low_end_1 = pd.Timestamp(year=yr, month=mo, day=dy, tz='UTC', **low_end_hour_1)
    #
    #         high_start_1 = pd.Timestamp(year=yr, month=mo, day=dy, tz='UTC', **high_start_hour_1)
    #         high_end_1 = pd.Timestamp(year=yr, month=mo, day=dy, tz='UTC', **high_end_hour_1)
    #
    #         # --------------------------------------------------------------------------------------------------
    #
    #         low_start_2 = pd.Timestamp(year=yr, month=mo, day=dy, tz='UTC', **low_start_hour_2)
    #         low_end_2 = pd.Timestamp(year=yr, month=mo, day=dy, tz='UTC', **low_end_hour_2)
    #
    #         high_start_2 = pd.Timestamp(year=yr, month=mo, day=dy, tz='UTC', **high_start_hour_2)
    #         high_end_2 = pd.Timestamp(year=yr, month=mo, day=dy, tz='UTC', **high_end_hour_2)
    #
    #         today_hist = alaska.history(interval='1m', start=today_minus_days.date(), end=today_minus_days.date() + timedelta(days=1))
    #         low_hist_1 = today_hist[(today_hist.index >= low_start_1) & (today_hist.index <= low_end_1)]
    #         high_hist_1 = today_hist[(today_hist.index >= high_start_1) & (today_hist.index <= high_end_1)]
    #         low_hist_2 = today_hist[(today_hist.index >= low_start_2) & (today_hist.index <= low_end_2)]
    #         high_hist_2 = today_hist[(today_hist.index >= high_start_2) & (today_hist.index <= high_end_2)]
    #
    #         high_high_1 = high_hist_1['Close'].max()
    #         low_low_1 = low_hist_1['Close'].min()
    #
    #         diff_1 = high_high_1 - low_low_1
    #         pct_1 = ((high_high_1 - low_low_1) / low_low_1) * 100
    #
    #         shares = start_investment / (low_low_1 / price_adjust)
    #         start_investment = shares * (high_high_1 * price_adjust)
    #
    #         # ----------------------------------------------------------------------------------------------------------
    #
    #         high_high_2 = high_hist_2['Close'].max()
    #         low_low_2 = low_hist_2['Close'].min()
    #
    #         diff_2 = high_high_2 - low_low_2
    #         pct_2 = ((high_high_2 - low_low_2) / low_low_2) * 100
    #
    #         shares = start_investment / (low_low_2 / price_adjust)
    #         start_investment = shares * (high_high_2 * price_adjust)
    #
    #         values['Date'].append(today_minus_days.date())
    #         values['Low_1'].append(f'${low_low_1:,.2f}')
    #         values['High_1'].append(f'${high_high_1:,.2f}')
    #         values['Diff_1'].append(f'${diff_1:,.2f}')
    #         values['Pct_1'].append(f'{pct_1:.2f}%')
    #         values['_____'].append('')
    #         values['Low_2'].append(f'${low_low_2:,.2f}')
    #         values['High_2'].append(f'${high_high_2:,.2f}')
    #         values['Diff_2'].append(f'${diff_2:,.2f}')
    #         values['Pct_2'].append(f'{pct_2:.2f}%')
    #         values['-----'].append('')
    #         values['Invest'].append(f'${start_investment:,.2f}')
    #
    #         # print()
    #         # print(today_minus_days)
    #         # print(f'Lowest: ${low_low_1:,.2f}')
    #         # print(f'High: ${high_high_1:,.2f}')
    #         # print(f'Difference: ${diff_1:,.2f}')
    #         # print(f'Percent: {pct_1:.2f}%')
    #         # print(f'Investment: ${start_investment_1:,.2f}')
    #
    #     today_minus_days = today_minus_days + timedelta(days=1)
    #
    # df = pd.DataFrame(data=values)
    #
    # start_money = float(df['Invest'][0].replace('$', '').replace(',', ''))
    # end_money = float(df['Invest'][len(df) - 1].replace('$', '').replace(',', ''))
    #
    # pct_gain = ((end_money - start_money) / start_money) * 100
    #
    # annual_gain = (1 + (pct_gain / 100)) ** 12
    #
    # print(df.to_string())
    # print(f'\nPercent Gained: {pct_gain:.2f}%')
    # print(f'\nIf maintain {pct_gain:.2f}% per month, annual percent would be {annual_gain * 100:.2f}%')
    # print(f"resulting in an initial investment of {df['Invest'][0]} turining into ${start_money * annual_gain:,.2f}")




    # alaska = yf.Ticker(ticker='BTC-USD')
    # print(alaska)
    # #print(alaska.info)
    # hist = alaska.history(period='1mo', interval='1min')
    # print(hist.to_string())
    #
    # hist_vals = hist.to_dict('list')
    # plt.scatter(x=list(hist.index), y=hist_vals['Close'])
    # plt.show()


    from binance import Client


    cli = Client()
    x = cli.get_historical_klines(symbol='BTC-USD', interval='1min', limit=10_000)
    print(x)