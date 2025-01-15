import json
import websocket
import requests

#
#     def _construct_api_url(self) -> str:
#         return self.twelve_data_api_url.format(ticker_symbols=','.join(self.ticker_symbols), interval=self.interval, api_key=self.api_key)
#
#     def api_request(self) -> dict:
#         url = self._construct_api_url()
#         r = requests.get(url)
#         data = r.json()
#         return data
#
#
#
def f_dict(d, t=0):
    tb = '\t' * t
    for k, v in d.items():
        if isinstance(v, dict):
            print(f'{tb}{k}' + '{')
            f_dict(v, t + 1)
        else:
            print(f'{tb}{k}: {v}')
    print('}')


#twelve_data_api_url = 'https://api.twelvedata.com/time_series?symbol={ticker_symbols}&interval={interval}&apikey={api_key}'


class Getter(websocket.WebSocketApp):

    twelve_data_websocket_url = 'wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}'

    def __init__(self, ticker_symbol: str, exchange: str):
        self.api_key = '6ffbaf5432db4c7993c13f09535c46b0'
        self.ticker_symbol = ticker_symbol
        self.exchange = exchange
        super(Getter, self).__init__(
            url=self.twelve_data_websocket_url.format(api_key=self.api_key),
        )
        self.on_open = self._on_open
        self.on_close = self._on_close
        self.on_message = self._on_message
        self.on_error = self._on_error

    def _on_open(self, *args, **kwargs):
        print(f'\n{self.__class__.__name__}: ON_OPEN')
        print(f'{args=}')
        print(f'{kwargs=}')
        subscribe = {
            'action': 'subscribe',
            'params': {
                'symbols': [{
                    'symbol': self.ticker_symbol,
                    'mic_code': 'XNYS'
                    #'exchange': self.exchange
                }]
            }
        }
        print(f'Subscription Header: {subscribe}')
        self.send(json.dumps(subscribe))

    def _on_close(self, *args, **kwargs):
        print(f'\n{self.__class__.__name__}: ON_CLOSE')
        print(f'{args=}')
        print(f'{kwargs=}')

    def _on_message(self, *args, **kwargs):
        print(f'\n{self.__class__.__name__}: ON_MESSAGE')
        print(f'{args[1]=}')

    def _on_error(self, *args, **kwargs):
        print(f'\n{self.__class__.__name__}: ON_ERROR')
        print(f'{args=}')
        print(f'{kwargs=}')








if __name__ == '__main__':

    g = Getter(ticker_symbol='ALK', exchange='NYSE')
    g.run_forever()

    # async def listen():
    #     url = 'wss://ws.twelvedata.com/v1/quotes/price?apikey=YOUR_API_KEY'

    # twelve_data_api_url = 'https://api.twelvedata.com/stocks'
    # r = requests.get(twelve_data_api_url)
    # data = r.json()
    # with open('stock.txt', 'w') as f:
    #     for key in data['data']:
    #         if key['symbol'].startswith('ALK'):
    #             print(key)
    #         #f.write(f'{key}\n')