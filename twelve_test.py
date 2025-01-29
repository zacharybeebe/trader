


if __name__ == '__main__':
    
    from trader.backend.data.engine import Engine
    from trader.backend.trading.robinhood import *
    from trader.backend.trading.strategy.strategies import *
    from trader.backend.trading.trading_manager import TradingManager
    from trader.backend.data.getters import RobinHoodGetter





    # x = pd.DataFrame({
    #     'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    # })

    # x['c'] = x['a'].pct_change(periods=5) * 100
    # print(x)


    # rhg = RobinHoodGetter()
    # xx = rhg.get_account()
    # print(xx)
    

    rh = Robinhood()
    z = rh.account
    x = rh.all_holdings
    y = rh.order_history(
        'DOGE',
        start=datetime(year=2020, month=12, day=1),
        side='sell',
        order_type='market',
    )
    s = rh.estimated_both('DOGE', 15000)
    ba = rh.best_bid_ask('DOGE')
    aa = rh.order_history_single(client_order_id='a219264d-8c2d-4bbd-a449-259d0f6492a9')
    print(z)
    print()
    print(x)
    print()
    for zzz in y:
        debug_print_dict(zzz)
    print(s)
    print(ba)
    debug_print_dict(aa)

    # idd = '6797b7f6-5a94-41f4-aafe-ab3a6f0a5dcf
    # client_order_id = 'a219264d-8c2d-4bbd-a449-259d0f6492a9'

    # xx = rh.qty_at_estimated_bid(
    #     'DOGE',
    #     1000
    # )
    # print(xx)
    # # debug_print_dict(xx)

    # print(rh.__private_key)

    # orders = rh.order_history(start=datetime(year=2025, month=1, day=6))
    # print(orders)
    # debug_print_dict(xx)
    # # for i in xx:
    # #     print(i)


    # CoinMarketCapWebSocket.run_client(url=CoinMarketCapWebSocket.PRICE_URL)


    # # p, _ = rhgem.get_pubticker('dogeusd')
    # p, _ = rhgem.get_pubticker('dogeusd', jsonify=True)
    # print(f'{p=}')
    # # print(f'{p.content=}')
    # # p_json = p.json()

    # # print(f'{p_json=}')
    # # print(p_json['volume']['timestamp'])
    # # dt = datetime.fromtimestamp(p_json['volume']['timestamp'] / 1000)
    # # print(f'{dt=}')


    # engine = Engine(database_type='sqlite', engine_echo=False)
    # trx = engine.start_transaction()

    # ticker = 'DOGE'
    # interval = '5m'

    # df = engine.data(ticker=ticker, interval=interval, start=None, end=None, transaction=trx)
    # print(df)
    # print(f'{len(df)=}')

    # next_dt = None
    # missing_dt = []
    # for i, idx in enumerate(df.index):
    #     current_dt = df.at[idx, 'dt']
    #     if next_dt is not None and current_dt != next_dt:
    #         previous_dt = df['dt'].iloc[i - 1]
    #         missing_dt.append({
    #             'index': idx,
    #             'previous_dt': previous_dt.strftime('%Y-%m-%d %H:%M:%S'),
    #             'current_dt': current_dt.strftime('%Y-%m-%d %H:%M:%S'),
    #             'next_dt': next_dt.strftime('%Y-%m-%d %H:%M:%S'),
    #             'delta_actual': pretty_time((current_dt - previous_dt).total_seconds())
    #         })
    #     next_dt = get_next_interval_dt(interval=interval, dt=df.at[idx, 'dt'])
    # print(f'{len(missing_dt)=}')
    # for mdt in missing_dt:
    #     debug_print_dict(mdt)

        



    # tb = TraderBot(
    #     ticker_symbol='DOGE/USD',
    #     interval='1min',

    # )

    # data = tb.send_request()
    # debug_print_dict(data['values'][0])
    