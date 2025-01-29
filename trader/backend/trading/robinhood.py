import base64
import datetime as dt
import json
import requests
import traceback

from nacl.signing import SigningKey

from .strategy.strategies import *





class Robinhood(object):
    __env = FromEnv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))
    __color_code = 36

    # Constructor ####################################################################################################################################
    def __init__(self):
        self.__api_key = self.__env.get('ROBINHOOD_API_KEY')
        self.__public_key = self.__env.get('ROBINHOOD_API_PUBLIC')
        self.__private_key = self._private_signing_key()
        self.base_url = 'https://trading.robinhood.com'

        self.order_info = {
            'market_buy': {},
            'market_sell': {},
            'limit_buy': {},
            'limit_sell': {},
            'stop_loss_buy': {},
            'stop_loss_sell': {},
            'stop_limit_buy': {},
            'stop_limit_sell': {}
        }
    
    # Private Methods ################################################################################################################################
    def _auth_headers(self, request_method: Literal['GET', 'POST'], url_suffix: str, body: Optional[dict] = None) -> dict:
        if body is None:
            body = ''
        else:
            body = json.dumps(body)
        ts = self._current_timestamp()
        message_to_sign = f'{self.__api_key}{ts}{url_suffix}{request_method}{body}'
        signed = self.__private_key.sign(message_to_sign.encode('utf-8'))
        return {
            'x-api-key': self.__api_key,
            'x-signature': base64.b64encode(signed.signature).decode('utf-8'),
            'x-timestamp': str(ts)
        }
    
    @staticmethod
    def _current_timestamp() -> int:
        return int(dt.datetime.now(tz=dt.timezone.utc).timestamp())
    
    def _estimated_price(self, symbol: str, side: Literal['bid', 'ask', 'both'], asset_quantity: Union[float, list[float]] = 1.0) -> Optional[float]:
        symbol = self._format_symbol(symbol, as_pair=True)
        if isinstance(asset_quantity, (list, tuple)):
            asset_quantity = ','.join([str(qty) for qty in asset_quantity])
        else:
            asset_quantity = str(asset_quantity)
        query_params = self._query_params_key_value(
            symbol=symbol,
            side=side,
            quantity=asset_quantity
        )
        response, _ = self._request(
            request_method='GET',
            url_suffix=f'/api/v1/crypto/marketdata/estimated_price/{query_params}'
        )
        return response
    
    @staticmethod
    def _format_symbol(symbol: str, as_pair: bool = True) -> str:
        symbol = symbol.upper()
        if as_pair and not symbol.endswith('-USD'):
            return f'{symbol}-USD'
        elif not as_pair and symbol.endswith('-USD'):
            return symbol[:-4]
        else:
            return symbol
    
    def _place_order(
            self,
            side: Literal['buy', 'sell'],
            order_type: Literal['market', 'limit', 'stop_limit', 'stop_loss'],
            symbol: str,
            asset_quantity: Optional[float] = None,
            quote_amount: Optional[float] = None,
            limit_price: float = 0,
            stop_price: float = 0,
            time_in_force: Literal['gtc', 'ioc', 'fok'] = 'gtc',
    ) -> Optional[dict]:
        side = side.lower()
        order_type = order_type.lower()
        order_config = {'asset_quantity': asset_quantity}
        if order_type != 'market':
            order_config['quote_amount'] = quote_amount
            order_config['time_in_force'] = time_in_force

            if order_type == 'limit':
                order_config['limit_price'] = limit_price
            elif order_type == 'stop_loss':
                order_config['stop_price'] = stop_price
            else:
                order_config['limit_price'] = limit_price
                order_config['stop_price'] = top_price

            if isna(order_config['asset_quantity']) or asset_quantity == 0:
                order_config.pop('asset_quantity')
            else:
                order_config.pop('quote_amount')
        
        if 'asset_quantity' in order_config:
            # Robinhood only allows 8 decimal places of precision
            order_config['asset_quantity'] = round(order_config['asset_quantity'], 8)        
        
        order_body = {
            'client_order_id': generate_uuid(),
            'symbol': self._format_symbol(symbol, as_pair=True),
            'side': side,
            'type': order_type,
            f'{order_type}_order_config': order_config
        }
        debug_print_dict(order_body)
        response, status = self._request(
            request_method='POST',
            url_suffix='/api/v1/crypto/trading/orders/',
            body=order_body
        )
        if str(status).startswith('2'):
            order_body['order_info'] = response
            self.order_info[f'{order_type}_{side}'][order_body['client_order_id']] = order_body
        return response

    @classmethod
    def _private_signing_key(cls) -> SigningKey:
        private_key_seed = base64.b64decode(cls.__env.get('ROBINHOOD_API_PRIVATE'))
        return SigningKey(private_key_seed)
    
    @classmethod
    def _prterror(cls, message: str) -> None:
        prtcolor(message, color_code=cls.__color_code, prefix=f'[{cls.__name__} ERROR]')
    
    @classmethod
    def _prtinfo(cls, message: str) -> None:
        prtcolor(message, color_code=cls.__color_code, prefix=f'[{cls.__name__} INFO]')
    
    @staticmethod
    def _query_params_same_key(key: str, *params: Optional[str]) -> str:
        if not params:
            return ''
        args = [f'{key}={param}' for param in params]
        return '?' + '&'.join(args)
    
    @staticmethod
    def _query_params_key_value(**key_value_pairs) -> str:
        if not key_value_pairs:
            return ''
        args = [f'{key}={value}' for key, value in key_value_pairs.items()]
        return '?' + '&'.join(args)
    
    def _request(self, request_method: Literal['GET', 'POST'], url_suffix: str, body: Optional[dict] = None) -> tuple[Optional[dict], Optional[int]]:
        if not url_suffix.startswith('/'):
            url_suffix = '/' + url_suffix
        headers = self._auth_headers(request_method, url_suffix, body)
        full_url = self.base_url + url_suffix
        print(f'{full_url=}')
        response = {}
        try:
            if request_method == 'GET':
                response = requests.get(full_url, headers=headers)
            elif request_method == 'POST':
                response = requests.post(full_url, headers=headers, json=body)
            else:
                raise ValueError(f'Invalid request method: {request_method}')
            print(f'{response=}')
            print(f'{response.status_code=}')
            return response.json(), response.status_code
        except requests.RequestException as e:
            self._prterror(f"Error making API request: {e}")
            self._prterror(traceback.format_exc())
            return None, None
        except Exception as e:
            raise e

    # Order Methods #################################################################################################################################
    def cancel_order(self, order_id: str) -> Optional[dict]:
        response, status = self._request(
            request_method='POST',
            url_suffix=f'/api/v1/crypto/trading/orders/{order_id}/cancel/'
        )
        if response is not None and str(status).startswith('2'):
            for order_type in self.order_info:
                if order_id in self.order_info[order_type]:
                    self.order_info[order_type].pop(order_id)
                    break
        return response

    def limit_buy(
            self, 
            symbol: str, 
            limit_price: float, 
            asset_quantity: Optional[float] = None,
            quote_amount: Optional[float] = None,
            time_in_force: Literal['gtc', 'ioc', 'fok'] = 'gtc'
        ) -> Optional[dict]:
        """
        Will place a limit buy order for the given symbol.\n\n

        A limit buy order is an order that will buy an asset when the price has reached or fallen 
        below a certain price, the "limit_price".\n\n

        The amount of asset to buy can be expressed in one of two ways, either by the "asset_quantity" or
        the "quote_amount", one of these arguments must be provided. If both are provided then the "asset_quantity"
        will be used. The Robinhood API only allows one of these arguments to be passed in the POST body. The
        "asset_quantity" is the amount of the asset (shares) to buy, while the "quote_amount" is the amount of cash to spend.
        """
        return self._place_order(
            side='buy',
            order_type='limit',
            symbol=symbol,
            asset_quantity=asset_quantity,
            quote_amount=quote_amount,
            limit_price=limit_price,
            time_in_force=time_in_force
        )

    def limit_sell(
            self, 
            symbol: str, 
            limit_price: float, 
            asset_quantity: Optional[float] = None,
            quote_amount: Optional[float] = None,
            time_in_force: Literal['gtc', 'ioc', 'fok'] = 'gtc'
        ) -> Optional[dict]:
        """
        Will place a limit sell order for the given symbol.\n\n

        A limit sell order is an order that will sell an asset when the price has reached or exceeded 
        a certain price, the "limit_price".\n\n

        The amount of asset to sell can be expressed in one of two ways, either by the "asset_quantity" or
        the "quote_amount", one of these arguments must be provided. If both are provided then the "asset_quantity"
        will be used. The Robinhood API only allows one of these arguments to be passed in the POST body. The
        "asset_quantity" is the amount of the asset (shares) to sell, while the "quote_amount" is the amount of cash gain.
        """
        return self._place_order(
            side='sell',
            order_type='limit',
            symbol=symbol,
            asset_quantity=asset_quantity,
            quote_amount=quote_amount,
            limit_price=limit_price,
            time_in_force=time_in_force
        )

    def market_buy(self, symbol: str, asset_quantity: Optional[float] = None, quote_amount: Optional[float] = None) -> Optional[dict]:
        """
        Will place a market buy order for the given symbol.\n\n

        A market buy order is an order that will buy a certain asset at the current market price.\n\n

        The amount of asset to buy can be expressed in one of two ways, either by the "asset_quantity" or
        the "quote_amount", one of these arguments must be provided. The "asset_quantity" is the amount of the 
        asset (shares/coins) to buy, while the "quote_amount" is the amount of cash to spend. If both arguments are
        provided then only the "asset_quantity" will be used.\n\n 
        
        *NOTE* that the Robinhood API only allows "asset_quantity" for market orders, so if the "quote_amount" is 
        provided, then the asset quantity will be estimated using the qty_at_estimated_bid() method; however markets 
        can change rapidly, so the estimated quantity may not reflect the actual quantity by the time the order is placed.
        """
        if isna(asset_quantity) or asset_quantity == 0:
            asset_quantity = self.qty_at_estimated_bid(symbol=symbol, quote_amount=quote_amount)
        return self._place_order(
            side='buy',
            order_type='market',
            symbol=symbol,
            asset_quantity=asset_quantity
        )

    def market_sell(self, symbol: str, asset_quantity: Optional[float] = None, quote_amount: Optional[float] = None) -> Optional[dict]:
        """
        Will place a market sell order for the given symbol.\n\n

        A market sell order is an order that will sell a certain asset at the current market price.\n\n

        The amount of asset to sell can be expressed in one of two ways, either by the "asset_quantity" or
        the "quote_amount", one of these arguments must be provided. The "asset_quantity" is the amount of the 
        asset (shares/coins) to sell, while the "quote_amount" is the amount of cash to gain. If both arguments are
        provided then only the "asset_quantity" will be used.\n\n 
        
        *NOTE* that the Robinhood API only allows "asset_quantity" for market orders, so if the "quote_amount" is 
        provided, then the asset quantity will be estimated using the qty_at_estimated_ask() method; however markets 
        can change rapidly, so the estimated quantity may not reflect the actual quantity by the time the order is placed.    
        """
        if isna(asset_quantity) or asset_quantity == 0:
            asset_quantity = self.qty_at_estimated_ask(symbol=symbol, quote_amount=quote_amount)
        return self._place_order(
            side='sell',
            order_type='market',
            symbol=symbol,
            asset_quantity=asset_quantity
        )
    
    def stop_loss_buy(
            self, 
            symbol: str, 
            stop_price: float, 
            asset_quantity: Optional[float] = None,
            quote_amount: Optional[float] = None,
            time_in_force: Literal['gtc', 'ioc', 'fok'] = 'gtc'
        ) -> Optional[dict]:
        return self._place_order(
            side='buy',
            order_type='stop_loss',
            symbol=symbol,
            asset_quantity=asset_quantity,
            quote_amount=quote_amount,
            stop_price=stop_price,
            time_in_force=time_in_force
        )

    def stop_loss_sell(
            self, 
            symbol: str, 
            stop_price: float, 
            asset_quantity: Optional[float] = None,
            quote_amount: Optional[float] = None,
            time_in_force: Literal['gtc', 'ioc', 'fok'] = 'gtc'
        ) -> Optional[dict]:
        return self._place_order(
            side='sell',
            order_type='stop_loss',
            symbol=symbol,
            asset_quantity=asset_quantity,
            quote_amount=quote_amount,
            stop_price=stop_price,
            time_in_force=time_in_force
        )

    def stop_limit_buy(
            self, 
            symbol: str, 
            limit_price: float, 
            stop_price: float,  
            asset_quantity: Optional[float] = None,
            quote_amount: Optional[float] = None,
            time_in_force: Literal['gtc', 'ioc', 'fok'] = 'gtc'
        ) -> Optional[dict]:
        """
        This will place a stop limit buy order for the given symbol.\n\n

        A stop limit buy order is an order that will buy an asset when the price has reached or fallen 
        below a certain price, the "limit_price", but only if the price has first fallen from a certain price, 
        the "stop_price".\n\n

        The amount of asset to buy can be expressed in one of two ways, either by the "asset_quantity" or
        the "quote_amount", one of these arguments must be provided. If both are provided then the "asset_quantity"
        will be used. The Robinhood API only allows one of these arguments to be passed in the POST body. The
        "asset_quantity" is the amount of the asset (shares) to buy, while the "quote_amount" is the amount of cash to spend.
        """
        return self._place_order(
            side='buy',
            order_type='stop_limit',
            symbol=symbol,
            asset_quantity=asset_quantity,
            quote_amount=quote_amount,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force
        )

    def stop_limit_sell(
            self, 
            symbol: str, 
            limit_price: float, 
            stop_price: float, 
            asset_quantity: Optional[float] = None,
            quote_amount: Optional[float] = None,
            time_in_force: Literal['gtc', 'ioc', 'fok'] = 'gtc'
        ) -> Optional[dict]:
        """
        This will place a stop limit sell order for the given symbol.\n
        A stop limit sell order is an order that will sell an asset when the price has reached or exceeded
        a certain price, the "limit_price", but only if the price has first increased from a certain price, 
        the "stop_price".\n\n

        The amount of asset to sell can be expressed in one of two ways, either by the "asset_quantity" or
        the "quote_amount", one of these arguments must be provided. If both are provided then the "asset_quantity"
        will be used. The Robinhood API only allows one of these arguments to be passed in the POST body. The
        "asset_quantity" is the amount of the asset (shares) to sell, while the "quote_amount" is the amount of cash to gain.
        """
        return self._place_order(
            side='sell',
            order_type='stop_limit',
            symbol=symbol,
            asset_quantity=asset_quantity,
            quote_amount=quote_amount,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force
        )
    
    # Info Methods ##################################################################################################################################
    def best_bid_ask(self, *symbols: Optional[str]) -> Optional[dict]:
        if symbols:
            symbols = [self._format_symbol(symbol, as_pair=True) for symbol in symbols]
        query_params = self._query_params_same_key('symbol', *symbols)
        response, _ = self._request(
            request_method='GET', 
            url_suffix=f'/api/v1/crypto/marketdata/best_bid_ask/{query_params}'
        )
        if response is not None:
            bid_ask = {}
            for item in response['results']:
                bid_ask[item['symbol']] = item
            debug_print_dict(bid_ask)
            return bid_ask
        else:
            return None

    def estimated_bid(self, symbol: str, asset_quantity: Union[float, list[float]] = 1.0) -> Optional[dict]:
        response = self._estimated_price(symbol, 'bid', asset_quantity)
        if response is not None:
            return response['results'][0]
        else:
            return None
    
    def estimated_ask(self, symbol: str, asset_quantity: Union[float, list[float]] = 1.0) -> Optional[dict]:
        response = self._estimated_price(symbol, 'ask', asset_quantity)
        if response is not None:
            return response['results'][0]
        else:
            return None
    
    def estimated_both(self, symbol: str, asset_quantity: Union[float, list[float]] = 1.0) -> Optional[dict]:
        response = self._estimated_price(symbol, 'both', asset_quantity)
        if response is not None:
            bid_ask = {}
            for item in response['results']:
                bid_ask[item['side']] = item
            return bid_ask
        else:
            return None    

    def export_order_history(
            self, 
            *symbols: Optional[str], 
            filename: Optional[str] = None, 
            directory: Optional[str] = None,
            start: Optional[dt.datetime] = None, 
            end: Optional[dt.datetime] = None,
            side: Optional[Literal['buy', 'sell']] = None,
            order_type: Optional[Literal['market', 'limit', 'stop_limit', 'stop_loss']] = None
        ) -> None:
        order_history = self.order_history(
            *symbols, 
            start=start, 
            end=end,
            side=side,
            order_type=order_type
        )
        if order_history is not None:
            if filename is None:
                filename = 'order_history.json'
            elif not filename.endswith('.json'):
                filename += '.json'
            if directory is None:
                directory = os.getcwd()
            with open(os.path.join(directory, filename), 'w') as f:
                json.dump(order_history, f, indent=4)
    
    def holdings(self, *symbols: Optional[str]) -> Optional[list]:
        symbols = [self._format_symbol(symbol, as_pair=False) for symbol in symbols]
        query_params = self._query_params_same_key('asset_code', *symbols)
        response, _ = self._request(
            request_method='GET', 
            url_suffix=f'/api/v1/crypto/trading/holdings/{query_params}'
        )
        if response is not None:
            return response['results']
        else:
            return None
    
    def order_history(
            self, 
            *symbols: Optional[str], 
            start: Optional[dt.datetime] = None, 
            end: Optional[dt.datetime] = None,
            side: Optional[Literal['buy', 'sell']] = None,
            order_type: Optional[Literal['market', 'limit', 'stop_limit', 'stop_loss']] = None
        ) -> Optional[list]:
        symbols = [self._format_symbol(symbol, as_pair=True) for symbol in symbols]
        response, _ = self._request(
            request_method='GET',
            url_suffix='/api/v1/crypto/trading/orders/'
        )
        if response is not None:
            if start is not None:
                start = start.replace(tzinfo=dt.timezone(dt.timedelta(days=-1, seconds=68400)))
            if end is not None:
                end = end.replace(tzinfo=dt.timezone(dt.timedelta(days=-1, seconds=68400)))

            orders = []
            for order in response['results']:
                do_add= True
                if symbols and order['symbol'] not in symbols:
                    do_add = False
                if start and datetime.fromisoformat(order['created_at']) < start:
                    do_add = False
                if end and datetime.fromisoformat(order['created_at']) > end:
                    do_add = False
                if side and order['side'] != side:
                    do_add = False
                if order_type and order['type'] != order_type:
                    do_add = False
                if do_add:
                    orders.append(order)
            return orders
        else:
            return None
    
    def order_history_single(self, client_order_id: str) -> Optional[dict]:
        order_history = self.order_history()
        if order_history is not None and len(order_history) > 0:
            for order in order_history:
                if order['client_order_id'] == client_order_id:
                    return order
        return None
    
    def qty_at_estimated_ask(self, symbol: str, quote_amount: float, drawdown: float = 0.985) -> Optional[float]:
        symbol = self._format_symbol(symbol, as_pair=True)
        response = self.best_bid_ask(symbol)
        if response is not None:
            ask_price = float(response[symbol]['bid_inclusive_of_sell_spread'])
            return quote_amount / ask_price * drawdown
        else:
            return None
    
    def qty_at_estimated_bid(self, symbol: str, quote_amount: float, drawdown: float = 0.985) -> Optional[float]:
        symbol = self._format_symbol(symbol, as_pair=True)
        response = self.best_bid_ask(symbol)
        if response is not None:
            bid_price = float(response[symbol]['ask_inclusive_of_buy_spread'])
            return quote_amount / bid_price * drawdown
        else:
            return None

    def trading_pairs(self, *symbols: Optional[str]) -> Optional[dict]:
        if symbols:
            symbols = [self._format_symbol(symbol, as_pair=True) for symbol in symbols]
        query_params = self._query_params_same_key('symbol', *symbols)
        response, _ = self._request(
            request_method='GET', 
            url_suffix=f'/api/v1/crypto/trading/trading_pairs/{query_params}'
        )
        if response is not None:
            trading_pairs = {}
            for trading_pair in response['results']:
                trading_pairs[trading_pair['asset_code']] = trading_pair
            return trading_pairs
        else:
            return None
        
    # Properties ####################################################################################################################################
    @property
    def account(self) -> Optional[dict]:
        response, _ = self._request(
            request_method='GET', 
            url_suffix='/api/v1/crypto/trading/accounts/'
        )
        return response
    
    @property
    def all_holdings(self) -> Optional[list]:
        return self.holdings()
    
    @property
    def all_trading_pairs(self) -> Optional[dict]:
        response, _ = self.trading_pairs()
        if response is not None:
            trading_pairs = {}
            for trading_pair in response['results']:
                trading_pairs[trading_pair['asset_code']] = trading_pair
            return trading_pairs
        else:
            return None
    
    @property
    def buying_power(self) -> Optional[float]:
        account = self.account
        if account is not None:
            return float(account['buying_power'])
        else:
            return None
        
    @property
    def all_order_history(self) -> Optional[dict]:
        return self.order_history()






