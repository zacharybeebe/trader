import asyncio
import json
import os
import websockets

from .robinhood import *
from ..data.engine import Engine


class TradingManager:
    COIN_MC_URL = 'wss://push.coinmarketcap.com/ws?device=web&client_source=coin_detail_page'
    COIN_MC_IDS = {
        1: 'BTC',
        2: 'LTC',
        74: 'DOGE',
        1027: 'ETH',
        1321: 'ETC',
    }
    COIN_MC_IDS_REV = {
        'BTC': 1,
        'LTC': 2,
        'DOGE': 74,
        'ETH': 1027,
        'ETC': 1321,
    }

    def __init__(self):
        pass

    def run(self):
        pass
    
    @classmethod
    def run_dummy_realtime_cryptos(
        cls,
        strategy: type[Strategy],
        cash: float,
        commission: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False,
    ) -> Broker:
        cryptos = {
            'DOGE': {
                'data': pd.DataFrame({'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}),
                'broker': None,
                'trade': None
            },
            # 'BTC': {
            #     'data': pd.DataFrame({'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}),
            #     'broker': None,
            #     'trade': None
            # },
            # 'ETH': {
            #     'data': pd.DataFrame({'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}),
            #     'broker': None,
            #     'trade': None
            # },
            # 'LTC': {
            #     'data': pd.DataFrame({'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}),
            #     'broker': None,
            #     'trade': None
            # },
            # 'ETC': {
            #     'data': pd.DataFrame({'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}),
            #     'broker': None,
            #     'trade': None
            # }
        }
        for coin in cryptos:
            cryptos[coin]['cash'] = cash / len(cryptos)
        ############################################################################################################################
        async def _run_trading():
            async with websockets.connect(cls.COIN_MC_URL) as ws:
                payload = json.dumps({
                    'method': 'RSUBSCRIPTION',
                    'params': [
                        'main-site@crypto_price_5s@{}@normal',
                        ','.join([str(cls.COIN_MC_IDS_REV[k]) for k in cls.COIN_MC_IDS_REV if k in cryptos])
                    ]
                })
                await ws.send(payload)
                # Initial handshake                
                await ws.recv()
                print('\nStarting Trading Loop\n')
                while True:
                    # Coin data
                    response = await ws.recv()
                    response = json.loads(response)
                    coin = cls.COIN_MC_IDS.get(response['d']['id'], None)
                    if coin is not None:
                        print(f'Coin: "{coin}"')
                        current_price = response['d']['p']
                        current_time = datetime.fromtimestamp(int(response['t']) / 1000)
                        if cryptos[coin]['broker'] is None:
                            cryptos[coin]['broker'] = Broker(
                                start_price=current_price,
                                start_date=current_time,
                                cash=cryptos[coin]['cash'],
                                commission=commission,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                restrict_sell_below_buy=restrict_sell_below_buy,
                                restrict_non_profitable=restrict_non_profitable,
                                close_end_position=False
                            )

                        cryptos[coin]['data'] = pd.concat([
                            cryptos[coin]['data'],
                            pd.DataFrame({
                                'date': [current_time],
                                'open': [np.nan],
                                'high': [np.nan],
                                'low': [np.nan],
                                'close': [current_price],
                                'volume': [np.nan],
                            })
                        ])
                        if len(cryptos[coin]['data']) > 20:
                            strtgy = strategy(
                                data=cryptos[coin]['data'], 
                                use_data_inplace=False,
                            )
                            previous_price = strtgy.data['close'].iloc[-2]
                            pct_change = ((current_price - previous_price) / previous_price) * 100
                            buy_signal = strtgy.data[strtgy.buy_column].iloc[-1]
                            sell_signal = strtgy.data[strtgy.sell_column].iloc[-1]
                            cryptos[coin]['trade'], cryptos[coin]['cash'] = cryptos[coin]['broker'].evaluate(
                                at_price=current_price,
                                at_date=current_time,
                                buy_signal=buy_signal,
                                sell_signal=sell_signal,
                                current_cash=cryptos[coin]['cash'],
                                current_trade=cryptos[coin]['trade'],
                            )

                            if cryptos[coin]['trade'] is None:
                                if len(cryptos[coin]['broker'].trades) > 0 and cryptos[coin]['broker'].trades[-1].closed == current_time:
                                    _last_trade = cryptos[coin]['broker'].trades[-1]
                                    display_dict = {
                                        'Current Time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                        'Current Price': f'${current_price:,.4f}',
                                        'Pct Change': f'{pct_change:,.2f}%',
                                        'Current Cash': f'${cryptos[coin]["cash"]:,.2f}',
                                        'Total Trades': f'{len(cryptos[coin]["broker"].trades):,.0f}',
                                        'PnL': f'${_last_trade.pnl:,.2f}',
                                        'Commssion': f'${_last_trade.commission:,.2f}',
                                        'Duration': pretty_time(_last_trade.duration.total_seconds()),
                                        'Buy Price': f'${_last_trade.buy_price:,.4f}',
                                        'Min Price': f'${_last_trade.min_price:,.4f}',
                                        'Max Price': f'${_last_trade.max_price:,.4f}',
                                        'Sell Price': f'${current_price:,.4f}',
                                    }
                                    table = tablefy_dict(display_dict, max_display_length=13)
                                    prtcolor('TRADE CLOSED!:\n' + table, color_code=32)
                            else:              
                                display_dict = {
                                    'Current Time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    'Current Price': f'${current_price:,.4f}',
                                    'Pct Change': f'{pct_change:,.2f}%',
                                    'Shares': f'{cryptos[coin]["trade"].shares:,.2f}',
                                    'Gross Value': f'${cryptos[coin]["trade"].gross_value(price=current_price):,.4f}',
                                    'Net Value': f'${cryptos[coin]["trade"].net_value(price=current_price):,.4f}',
                                    'Total Trades': f'{len(cryptos[coin]["broker"].trades):,.0f}',
                                    'Buy Price': f'${cryptos[coin]["trade"].buy_price:,.4f}',
                                }          
                                table = tablefy_dict(display_dict)        
                                if cryptos[coin]['trade'].opened == current_time:
                                    prtcolor('TRADE OPENED!:\n' + table, color_code=35)
                                else:
                                    prtcolor('IN POSITION:\n' + table, color_code=34)
                            print('\n')
        ############################################################################################################################
        print(f'Starting Realtime Dummy Trading at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'\t- {"Cryptos":25}{list(cryptos.keys())}')
        print(f'\t- {"Strategy":25}{strategy.__name__}')
        print(f'\t- {"Cash":25}${cash:,.2f}')
        print(f'\t- {"Commission":25}{commission:.3f}%')
        if stop_loss is not None:
            print(f'\t- {"Stop Loss":25}{stop_loss:.3f}%')
        else:
            print(f'\t- {"Stop Loss":25}{stop_loss}')
        if take_profit is not None:
            print(f'\t- {"Take Profit":25}{take_profit:.3f}%')
        else:
            print(f'\t- {"Take Profit":25}{take_profit}')
        print(f'\t- {"Restrict Sell Below Buy":25}{restrict_sell_below_buy}')
        print(f'\t- {"Restrict Non Profitable":25}{restrict_non_profitable}')
        try:
            asyncio.get_event_loop().run_until_complete(_run_trading())
        finally:        
            return cryptos
    
    @classmethod
    def run_dummy(
        cls,
        ticker: str,
        interval: str,
        engine: Engine, 
        strategy: type[Strategy],
        cash: float,
        commission: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        restrict_sell_below_buy: bool = False,
        restrict_non_profitable: bool = False,
    ) -> Broker:
        start_time = datetime.now()
        ticker_df = engine.data(ticker=ticker, interval=interval, start=None)
        init_price = ticker_df['close'].iloc[-1]
        init_dt = ticker_df['dt'].iloc[-1]
        broker = Broker(
            start_price=init_price,
            start_date=start_time,
            cash=cash,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit,
            restrict_sell_below_buy=restrict_sell_below_buy,
            restrict_non_profitable=restrict_non_profitable,
            close_end_position=False
        )
        current_cash = float(broker.initial_cash)
        current_trade = None
        print(f'Starting Dummy Trading at {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'\t- {"Ticker":25}{ticker}')
        print(f'\t- {"Interval":25}{interval}')
        print(f'\t- {"Strategy":25}{strategy.__name__}')
        print(f'\t- {"Cash":25}${cash:,.2f}')
        print(f'\t- {"Commission":25}{commission:.3f}%')
        if stop_loss is not None:
            print(f'\t- {"Stop Loss":25}{stop_loss:.3f}%')
        else:
            print(f'\t- {"Stop Loss":25}{stop_loss}')
        if take_profit is not None:
            print(f'\t- {"Take Profit":25}{take_profit:.3f}%')
        else:
            print(f'\t- {"Take Profit":25}{take_profit}')
        print(f'\t- {"Restrict Sell Below Buy":25}{restrict_sell_below_buy}')
        print(f'\t- {"Restrict Non Profitable":25}{restrict_non_profitable}')
        print(f'\t- {"Init Price":25}${init_price:,.4f}')
        print(f'\t- {"Init Time":25}{init_dt.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        cls.sleep_to_next_interval(interval=interval)
        while True:
            try:
                now = datetime.now()
                ticker_df = engine.data(ticker=ticker, interval=interval, start=None)
                strtgy = strategy(
                    data=ticker_df, 
                    use_data_inplace=False,
                    column_mappers={'dt': 'date'}
                )
                current_price = strtgy.data['close'].iloc[-1]
                previous_price = strtgy.data['close'].iloc[-2]
                pct_change = ((current_price - previous_price) / previous_price) * 100
                current_dt = strtgy.data.index[-1]
                buy_signal = strtgy.data[strtgy.buy_column].iloc[-1]
                sell_signal = strtgy.data[strtgy.sell_column].iloc[-1]
                current_trade, current_cash = broker.evaluate(
                    at_price=current_price,
                    at_date=current_dt,
                    buy_signal=buy_signal,
                    sell_signal=sell_signal,
                    current_cash=current_cash,
                    current_trade=current_trade,
                )

                if current_trade is None:
                    if len(broker.trades) > 0 and broker.trades[-1].closed == current_dt:
                        _last_trade = broker.trades[-1]
                        display_dict = {
                            'Current Time': current_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            'Current Price': f'${current_price:,.4f}',
                            'Pct Change': f'{pct_change:,.2f}%',
                            'Current Cash': f'${current_cash:,.2f}',
                            'Total Trades': f'{len(broker.trades):,.0f}',
                            'PnL': f'${_last_trade.pnl:,.2f}',
                            'Commssion': f'${_last_trade.commission:,.2f}',
                            'Duration': pretty_time(_last_trade.duration.total_seconds()),
                            'Buy Price': f'${_last_trade.buy_price:,.4f}',
                            'Min Price': f'${_last_trade.min_price:,.4f}',
                            'Max Price': f'${_last_trade.max_price:,.4f}',
                            'Sell Price': f'${current_price:,.4f}',
                        }
                        table = tablefy_dict(display_dict, max_display_length=13)
                        prtcolor('TRADE CLOSED!:\n' + table, color_code=32)
                    else:
                        display_dict = {
                            'Current Time': current_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            'Current Price': f'${current_price:,.4f}',
                            'Pct Change': f'{pct_change:,.2f}%',
                            'Current Cash': f'${current_cash:,.2f}',
                            'Total Trades': f'{len(broker.trades):,.0f}',
                        }
                        table = tablefy_dict(display_dict)
                        prtcolor('Not in position:\n' + table, color_code=34)
                else:
                    display_dict = {
                        'Now': now.strftime("%Y-%m-%d %H:%M:%S"),
                        'Current Time': current_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        'Current Price': f'${current_price:,.4f}',
                        'Pct Change': f'{pct_change:,.2f}%',
                        'Shares': f'{current_trade.shares:,.2f}',
                        'Gross Value': f'${current_trade.gross_value(price=current_price):,.2f}',
                        'Net Value': f'${current_trade.net_value(price=current_price):,.2f}',
                        'Total Trades': f'{len(broker.trades):,.0f}',
                        'Buy Price': f'${current_trade.buy_price:,.4f}',
                    }
                    table = tablefy_dict(display_dict)
                    if current_trade.opened == current_dt:
                        prtcolor('TRADE OPENED!:\n' + table, color_code=35)
                    else:
                        prtcolor('IN POSITION:\n' + table, color_code=35)
                print('\n')
                
                cls.sleep_to_next_interval(interval=interval)

            except KeyboardInterrupt:
                break

            except Exception as e:
                print(f'{e=}')
                print(traceback.format_exc())
                break
        
        end_time = datetime.now()
        broker.complete(
            end_price=current_price,
            end_date=end_time,
            current_cash=current_cash,
            current_trade=current_trade
        )
        return broker     
    
    @classmethod
    def run_doge_5m_waves(
            cls, 
            gain_pct: int = 8,
            retract_pct: int = 10,
            reset_low_pct: int = 15,
            reset_high_pct: int = 16
        ) -> None:
        rh = Robinhood()
        ticker = 'DOGE'
        interval = '5m'
        ###################################################################
        def get_cash_and_trade() -> tuple[float, Optional[dict]]:
            current_cash = rh.buying_power
            all_holdings = rh.all_holdings
            if all_holdings is not None and len(all_holdings) > 0:
                current_trade = rh.all_holdings[0]
            else:
                current_trade = None
            return current_cash, current_trade
        
        def get_last_order(side: Literal['buy', 'sell']) -> Optional[dict]:
            order_history = rh.order_history('DOGE', side=side)
            if order_history is not None and len(order_history) > 0:
                return order_history[0]
            return None
        
        ###################################################################        
        current_cash, current_trade = get_cash_and_trade()
        if current_trade is not None:
            in_position = True
            shares = float(current_trade['total_quantity'])
            sell_anchor = float(get_last_order(side='buy')['average_price'])
            buy_anchor = None
        else:
            in_position = False
            shares = 0
            buy_anchor = float(get_last_order(side='sell')['average_price'])
            sell_anchor = None

        buy_anchor_reset = False
        sell_anchor_reset = False

        init_estimated_prices = rh.best_bid_ask(ticker)
        if init_estimated_prices is not None:
            buy_price = float(estimated_prices[f'{ticker}-USD']['bid_inclusive_of_sell_spread'])
            sell_price = float(estimated_prices[f'{ticker}-USD']['ask_inclusive_of_buy_spread'])
        else:
            buy_price = None
            sell_price = None

        cls.sleep_to_next_interval(interval=interval)
        while True:
            try:
                now = datetime.now()
                estimated_prices = rh.best_bid_ask(ticker)
                if estimated_prices is not None:
                    current_buy_price = float(estimated_prices[f'{ticker}-USD']['bid_inclusive_of_sell_spread'])
                    current_sell_price = float(estimated_prices[f'{ticker}-USD']['ask_inclusive_of_buy_spread'])
                    if buy_price is not None:
                        buy_pct_change = ((current_buy_price - buy_price) / buy_price) * 100
                    else:
                        buy_pct_change = None
                    if sell_price is not None:
                        sell_pct_change = ((current_sell_price - sell_price) / sell_price) * 100
                    else:
                        sell_pct_change = None
                    buy_price = current_buy_price
                    sell_price = current_sell_price                    
                    table_dict = None
                    order_type = None
                    sub_order_type = None

                    # Do the inital buy at whatever price if not in position for the first iteration
                    if sell_anchor is None:
                        trade_results = rh.market_buy(ticker, quote_amount=current_cash)
                        while trade_results['state'] != 'filled':
                            trade_results = rh.order_history_single(client_order_id=trade_results['client_order_id'])                        
                        sell_anchor = float(trade_results['average_price'])
                        shares = float(trade_results['filled_asset_quantity'])
                        in_position = True
                        current_cash, current_trade = get_cash_and_trade()
                        order_type = 'Buy'
                        sub_order_type = 'Initial'
                        table_dict = {
                            'Order Type': order_type,
                            'Sub Order Type': sub_order_type,
                            'Current Time': now.strftime("%Y-%m-%d %H:%M:%S"),
                            'Actual Buy Price': f'${sell_anchor:,.4f}',
                            'Shares': f'{shares:,.1f}',
                            'Est Buy Price': f'${buy_price:,.4f}',
                            'Buy Pct Change': 'None' if buy_pct_change is None else f'{buy_pct_change:,.2f}%',
                            'Sell Pct Change': 'None' if sell_pct_change is None else f'{sell_pct_change:,.2f}%',
                            'Current Cash': f'${current_cash:,.2f}'
                        }
                    
                    # Examine if we should sell
                    elif in_position and sell_price >= sell_anchor * (1 + (gain_pct / 100)):
                        trade_results = rh.market_sell(ticker, asset_quantity=shares)
                        while trade_results['state'] != 'filled':
                            trade_results = rh.order_history_single(client_order_id=trade_results['client_order_id'])                        
                        buy_anchor = float(trade_results['average_price'])
                        shares = 0
                        in_position = False
                        current_cash, current_trade = get_cash_and_trade()
                        order_type = 'Sell'
                        if sell_anchor_reset:
                            sub_order_type = 'Reset'
                            sell_anchor_reset = False
                        else:
                            sub_order_type = 'Normal'
                        table_dict = {
                            'Order Type': order_type,
                            'Sub Order Type': sub_order_type,
                            'Current Time': now.strftime("%Y-%m-%d %H:%M:%S"),
                            'Actual Sell Price': f'${buy_anchor:,.4f}',
                            'Est Sell Price': f'${sell_price:,.4f}',
                            'Buy Pct Change': 'None' if buy_pct_change is None else f'{buy_pct_change:,.2f}%',
                            'Sell Pct Change': 'None' if sell_pct_change is None else f'{sell_pct_change:,.2f}%',
                            'Current Cash': f'${current_cash:,.2f}'
                        }

                    # Examine if we should buy
                    elif not in_position and buy_anchor is not None and buy_price <= buy_anchor * (1 - (retract_pct / 100)):
                        trade_results = rh.market_buy(ticker, quote_amount=current_cash)
                        while trade_results['state'] != 'filled':
                            trade_results = rh.order_history_single(client_order_id=trade_results['client_order_id'])                        
                        sell_anchor = float(trade_results['average_price'])
                        shares = float(trade_results['filled_asset_quantity'])
                        in_position = True
                        current_cash, current_trade = get_cash_and_trade()
                        order_type = 'Buy'
                        if buy_anchor_reset:
                            sub_order_type = 'Reset'
                            buy_anchor_reset = False
                        else:
                            sub_order_type = 'Normal'
                        table_dict = {
                            'Order Type': order_type,
                            'Sub Order Type': sub_order_type,
                            'Current Time': now.strftime("%Y-%m-%d %H:%M:%S"),
                            'Actual Buy Price': f'${sell_anchor:,.4f}',
                            'Shares': f'{shares:,.1f}',
                            'Est Buy Price': f'${buy_price:,.4f}',
                            'Buy Pct Change': 'None' if buy_pct_change is None else f'{buy_pct_change:,.2f}%',
                            'Sell Pct Change': 'None' if sell_pct_change is None else f'{sell_pct_change:,.2f}%',
                            'Current Cash': f'${current_cash:,.2f}'
                        }
                    
                    # Examine if we should reset the sell anchor
                    elif in_position and sell_price <= sell_anchor * (1 - (reset_low_pct / 100)):
                        sell_anchor = sell_price
                        sell_anchor_reset = True

                    # Examine if we should reset the buy anchor
                    elif not in_position and buy_price >= buy_anchor * (1 + (reset_high_pct / 100)):
                        buy_anchor = buy_price
                        buy_anchor_reset = True
                    

                    if table_dict is not None:
                        table = tablefy_dict(table_dict, max_display_length=17)
                        table_html = tablefy_dict_html(table_dict)
                        if order_type == 'Buy':
                            color_code = 35
                        else:
                            color_code = 32
                        prtcolor(f'{order_type} - {sub_order_type}\n' + table, color_code=color_code)
                        send_email(
                            to='z.beebe@yahoo.com',
                            subject=f'{order_type} - {sub_order_type}',
                            message=table_html,
                            message_contains_html=True
                        )

                cls.sleep_to_next_interval(interval=interval)

            except KeyboardInterrupt:
                break

            except Exception as e:
                print(f'{e=}')
                print(traceback.format_exc())
                break    

    
    @staticmethod
    def next_interval(interval: str) -> tuple[datetime, float]:
        now = datetime.now(tz=timezone.utc)
        next_interval_dt = get_next_interval_dt(interval=interval, dt=now)
        return next_interval_dt, (next_interval_dt - now).total_seconds()
    
    @classmethod
    def sleep_to_next_interval(cls, interval: str):
        _, sleep_time = cls.next_interval(interval=interval)
        time.sleep(sleep_time)