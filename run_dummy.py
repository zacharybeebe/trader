


if __name__ == '__main__':
    from trader.backend.data.engine import Engine
    from trader.backend.trading.strategy.strategies import *
    from trader.backend.trading.trading_manager import TradingManager
    
    
    "tickers = [DOGE, BTC, ETH, LTC]"
    " airline tickers = [UAL, DAL, AAL, LUV, ALK]"

    ticker = input('Enter Ticker [DOGE]: ').upper()
    if ticker == '':
        ticker = 'DOGE'
    interval = input('Enter Interval [1m]: ').lower()
    if interval == '':
        interval = '1m'
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


    engine = Engine(database_type='sqlite', engine_echo=False)
    # strategy = FibonacciRSI
    # strategy = FibSmaMod
    strategy = STDPeriods

    try:
        broker = TradingManager.run_dummy(
            ticker=ticker,
            interval=interval,
            engine=engine,
            strategy=strategy,
            cash=cash,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit,
            restrict_sell_below_buy=restrict_sell_below_buy,
            restrict_non_profitable=restrict_non_profitable,
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        print('\n\n' + ('-' * 100))
        print('End of Dummy Trading')
        print(broker)
        print('-' * 100)
        print('\n\n')


    

    