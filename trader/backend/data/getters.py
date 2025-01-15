import asyncio
import base64
import json
import nacl.signing
import requests
import traceback
import websockets

from ...utils.utils import *


class _BaseGetter(object):
    @staticmethod
    def _make_request(method: str, url: str, headers: dict = None, body: dict = None) -> Optional[dict]:
        try:
            response = {}
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=body, timeout=10)
            return response.json()
        
        except requests.RequestException as e:
            print(e)
            print(traceback.format_exc())
            return None



class TwelveDataGetter(_BaseGetter):
    _API_KEY = EnvReader.get('TWELVE_DATA_API_KEY')
    BASE_URL = 'https://api.twelvedata.com' #time_series?symbol={ticker_symbol}&interval={interval}&apikey={api_key}'

    
    @classmethod
    def get_price(cls, ticker: str, interval: T.Trade.INTERVAL) -> dict:
        final_ticker = ticker_check_pair(ticker=ticker, pair='USD', sep='/', lower=False)
        path = f'/price?symbol={final_ticker}&interval={pd_freq_from_interval(interval)}&apikey={cls._API_KEY}'
        url = cls.BASE_URL + path
        return cls._make_request(method='GET', url=url)

    @classmethod
    def get_quote(cls, ticker: str, interval: T.Trade.INTERVAL) -> dict:
        final_ticker = ticker_check_pair(ticker=ticker, pair='USD', sep='/', lower=False)
        path = f'/quote?symbol={final_ticker}&interval={pd_freq_from_interval(interval)}&apikey={cls._API_KEY}'
        url = cls.BASE_URL + path
        return cls._make_request(method='GET', url=url)
    
    @classmethod
    def get_time_series(cls, ticker: str, interval: T.Trade.INTERVAL) -> dict:
        final_ticker = ticker_check_pair(ticker=ticker, pair='USD', sep='/', lower=False)
        path = f'/time_series?symbol={final_ticker}&interval={pd_freq_from_interval(interval)}&apikey={cls._API_KEY}'
        url = cls.BASE_URL + path
        return cls._make_request(method='GET', url=url)



class RobinHoodGetter(_BaseGetter):
    _API_KEY = EnvReader.get('ROBINHOOD_API_KEY')
    _PRIVATE_KEY_B64 = EnvReader.get('ROBINHOOD_API_PRIVATE')
    _PUBLIC_KEY_B64 = EnvReader.get('ROBINHOOD_API_PUBLIC')
    _PRIVATE_KEY = nacl.signing.SigningKey(base64.b64decode(_PRIVATE_KEY_B64))
    # _PRIVATE_KEY = nacl.signing.SigningKey(base64.b64decode(_PUBLIC_KEY_B64))

    BASE_URL = 'https://trading.robinhood.com'

    
    @staticmethod
    def _get_query_params(key: str, *query_args) -> str:
        if not query_args:
            return ''
        params = []
        for arg in query_args:
            params.append(f'{key}={arg}')
        return '?' + '&'.join(params)


    @classmethod
    def _get_auth_header(cls, method: str, path: str, body: dict = '') -> dict:
        if body is None:
            f_body = ''
        else:
            f_body = json.dumps(body)
        timestamp = cls.get_timestamp_now()
        message_to_sign = f'{cls._API_KEY}{timestamp}{path}{method}{f_body}'
        print(f'{message_to_sign=}')
        signed_message = cls._PRIVATE_KEY.sign(message_to_sign.encode('utf-8'))
        print(f'{signed_message=}')
        headers = {
            'x-api-key': cls._API_KEY,
            'x-timestamp': str(timestamp),
            'x-signature': base64.b64encode(signed_message.signature).decode('utf-8')
        }
        print(f'{headers=}')
        return headers
    
    @classmethod
    def _make_api_request(cls, method: str, path: str, body: dict = None) -> Optional[dict]:
        timestamp = cls.get_timestamp_now()
        headers = cls._get_auth_header(method, path, body)
        url = cls.BASE_URL + path
        print(f'{url=}')
        return cls._make_request(
            method=method, 
            url=url, 
            headers=headers, 
            body=body
        )
        # try:
        #     response = {}
        #     if method == "GET":
        #         response = requests.get(url, headers=headers, timeout=10)
        #     elif method == "POST":
        #         if body is None:
        #             response = requests.post(url, headers=headers, timeout=10)
        #         else:
        #             response = requests.post(url, headers=headers, json=body, timeout=10)
        #     print(f'{response=}')
        #     return response.json()
        
        # except requests.RequestException as e:
        #     print(e)
        #     print(traceback.format_exc())
        #     # print(f"Error making API request: {e}")
        #     return None
    
    @classmethod
    def get_best_bid_ask(cls, *symbols: str) -> Any:
        final_symbols = [ticker_check_pair(ticker=s, pair='USD', sep='-', lower=False) for s in symbols]
        query_params = cls._get_query_params('symbol', *final_symbols)
        path = f'/api/v1/crypto/marketdata/best_bid_ask/{query_params}'
        return cls._make_api_request(method='GET', path=path)

    @classmethod
    def get_trading_pairs(cls, *symbols: str) -> dict:
        final_symbols = [ticker_check_pair(ticker=s, pair='USD', sep='-', lower=False) for s in symbols]
        query_params = cls._get_query_params('symbol', *final_symbols)
        path = f'/api/v1/crypto/trading/trading_pairs/{query_params}'
        return cls._make_api_request(method='GET', path=path)

    @staticmethod
    def get_timestamp_now() -> int:
        # return int(datetime.now(tz=timezone.utc).timestamp())
        return int(datetime.now().timestamp())
    
    def get_account(self) -> Any:
        path = "/api/v1/crypto/trading/accounts/"
        return self._make_api_request("GET", path)



class CoinMarketCapWebSocket(object):
    # PRICE_URL = 'wss://stream.coinmarketcap.com/price/latest'
    PRICE_URL = 'wss://push.coinmarketcap.com/ws?device=web&client_source=coin_detail_page'
    TRADE_URL = 'wss://stream.coinmarketcap.com/trade/latest'
    ORDERBOOK_URL = 'wss://stream.coinmarketcap.com/orderbook/latest'
    CANDLE_URL = 'wss://stream.coinmarketcap.com/candle/latest'

    IDS = {
        'BTC': 1,
        'LTC': 2,
        'DOGE': 74,
        'ETH': 1027,
        'ETC': 1321,
    }

    
    @classmethod
    def run_client(cls, url: str) -> None:
        async def _run_client():
            async with websockets.connect(url) as ws:
                payload = json.dumps({
                    'method': 'RSUBSCRIPTION',
                    'params': [
                        # 'main-site@crypto_price_15s@{}@detail',
                        'main-site@crypto_price_5s@{}@normal',
                        # '74,1,1027,2010,1839',
                        # '74',
                        ','.join([str(i) for i in cls.IDS.values()])
                    ]
                })
                await ws.send(payload)
                await ws.recv()
                while True:
                    response = await ws.recv()
                    response = json.loads(response)
                    debug_print_dict(response, add_types=True)
                    print(datetime.fromtimestamp(int(response['t']) / 1000))
        
        asyncio.get_event_loop().run_until_complete(_run_client())
