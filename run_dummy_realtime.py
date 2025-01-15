


if __name__ == '__main__':
    from trader.backend.data.engine import Engine
    from trader.backend.trading.strategy.strategies import *
    from trader.backend.trading.trading_manager import TradingManager
    

    try:
        cash = float(input('Enter Cash [25_000]: '))
    except ValueError:
        cash = 25_000
    try:
        commission = float(input('Enter Commission (whole percent) [0.46]: '))
    except ValueError:
        commission = 0.46
    try:
        stop_loss = float(input('Enter Stop Loss (whole percent) [None]: '))
    except ValueError:
        stop_loss = None
    try:
        take_profit = float(input('Enter Take Profit (whole percent) [None]: '))
    except ValueError:
        take_profit = None

    restrict_sell_below_buy = input('Restrict Sale when Sell Price is below Bought Price? (y/n): ').lower()
    if restrict_sell_below_buy == 'y':
        restrict_sell_below_buy = True
    else:
        restrict_sell_below_buy = False

    restrict_non_profitable = input('Restrict Sale when Sale is not profitable? (y/n): ').lower()
    if restrict_non_profitable == 'y':
        restrict_non_profitable = True
    else:
        restrict_non_profitable = False
    print('\n\n')

    # strategy = FibonacciRSI
    # strategy = FibSmaMod
    # strategy = PSLMacd
    # strategy = LinRegEmaSmaCross
    strategy = STDPeriods


    try:
        cryptos = TradingManager.run_dummy_realtime_cryptos(
            strategy=strategy,
            cash=cash,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit,
            restrict_sell_below_buy=restrict_sell_below_buy,
            restrict_non_profitable=restrict_non_profitable,
        )
    except KeyboardInterrupt:
        print('\nGot KeyboardInterrupt...\n')
        pass
    except Exception as e:
        print(f'\nGot Exception: {e}\n')
        print(traceback.format_exc())
        raise e
    finally:
        print('\nExiting Trading Loop\n')
        for coin in cryptos:
            print(f'Coin: "{coin}"')
            print(f'Len Data: {len(cryptos[coin]["data"])}')
            if cryptos[coin]['broker'] is not None:
                cryptos[coin]['broker'].complete(
                    end_price=cryptos[coin]['data']['close'].iloc[-1],
                    end_date=cryptos[coin]['data']['date'].iloc[-1],
                    current_cash=cryptos[coin]['cash'],
                    current_trade=cryptos[coin]['trade']
                )
                print(cryptos[coin]['broker'])
            else:
                print('No Trades')
            print('\n')
    


    

    