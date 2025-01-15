from typing import (
    Any, 
    Callable, 
    Dict,
    Generator, 
    Iterable,
    Literal, 
    List, 
    Optional, 
    Sequence, 
    Tuple, 
    Type, 
    Union
)

class T:
    class Trade:
        INTERVAL = Literal['1m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    class Strategy:
        MONTE_CARLO = Literal['normal', 'direct', 'brownian']