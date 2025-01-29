import inspect
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import smtplib
import shutil
import sys
import time
import traceback

from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from matplotlib.dates import DateFormatter
from matplotlib.ticker import StrMethodFormatter
from prophet import Prophet
from threading import Timer
from uuid import uuid4

from .typing import *



def apply_fibonacci_levels(close_price: pd.Series) -> tuple[pd.Series, pd.Series]:
    fib_levels = get_fibonaaci_levels(close_price.max(), close_price.min())
    ################################################################################
    def _calc_level(cp: float, level_type: str):
        if level_type == 'upper':
            return min([v for v in fib_levels.values() if v >= cp], default=np.nan)
        else:
            return max([v for v in fib_levels.values() if v <= cp], default=np.nan)
    ################################################################################
    upper_fib = close_price.apply(lambda x: _calc_level(x, 'upper'))
    lower_fib = close_price.apply(lambda x: _calc_level(x, 'lower'))
    return upper_fib, lower_fib


def chunker(sequence: Sequence, chunk_size: int) -> Generator:
    """
    Generator function that groups a sequence into chunks of a given size.
    Example:
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for c in chunker(x, 5):
            print(c)
            --> [1, 2, 3, 4, 5]
            --> [6, 7, 8, 9]
    """
    return (sequence[pos:pos + chunk_size] for pos in range(0, len(sequence), chunk_size))


def convert_numpy_value(value: Any, column: Optional[str] = None, null_zeroes_for_columns: Optional[list] = None) -> Any:
    """
    Possibly converts a numpy value (typically this value comes from a DataFrame) to its respective Python value

    :param value: The value to be possibly be converted
    :param column: The column name of the DataFrame in which the value resides - This is only used for nulling zero values
    :param null_zeroes_for_columns: A list of columns where zeros should be nullified
    :return: The converted value
    """
    if isna(value):
        v = None
    elif isinstance(value, (np.int8, np.int16, np.int32, np.int64)):
        v = int(value)
    elif isinstance(value, (np.float16, np.float32, np.float64)):
        v = float(value)
    elif isinstance(value, np.bool_):
        v = True if value else False
    elif isinstance(value, memoryview):
        v = bytes(value)
    else:
        v = value

    if null_zeroes_for_columns is not None:
        if column is not None:
            if column in null_zeroes_for_columns and v == 0:
                v = None
    return v


def dict_convert_numpy_values(dictionary: dict, nans_to_none: bool = True) -> dict:
    """
    Converts all numpy values in a dictionary to their respective Python values
    :param dictionary:              The dictionary to convert
    :return:    dict
    """
    ##############################################################################
    def recurisve_iterable(iterable):
        for i, v in enumerate(iterable):
            if isinstance(v, dict):
                iterable[i] = dict_convert_numpy_values(dictionary=v, nans_to_none=nans_to_none)
            elif isinstance(v, (list, set)):
                iterable[i] = recurisve_iterable(iterable=v)
            elif isinstance(v, tuple):
                iterable[i] = tuple(recurisve_iterable(iterable=list(v)))
            else:
                if not nans_to_none and pd.isna(v):
                    iterable[i] = v
                else:
                    iterable[i] = convert_numpy_value(value=v)
        return iterable
    ###############################################################################
    for key in dictionary:
        v = dictionary[key]
        if isinstance(v, dict):
            dictionary[key] = dict_convert_numpy_values(dictionary=v, nans_to_none=nans_to_none)
        elif isinstance(v, (list, set)):
            dictionary[key] = recurisve_iterable(iterable=v)
        elif isinstance(v, tuple):
            dictionary[key] = tuple(recurisve_iterable(iterable=list(v)))
        else:
            if not nans_to_none and pd.isna(v):
                dictionary[key] = v
            else:
                dictionary[key] = convert_numpy_value(value=v)
    return dictionary


def datetime_parse(dt: Optional[Union[str, datetime, date, pd.Timestamp]], nan_to_now: bool = False) -> Optional[datetime]:
    """
    This method will ensure that the given datetime "dt" returns as a Python datetime object.
    If the "dt" object isna, then None will be returned unless the "nan_to_now" flag is True, then datetime.now() will be returned.
    If the "dt" object is a datetime already, then it will be returned as is.
    If the "dt" object is a date, then it will be converted to a datetime object.
    If the "dt" object is a pd.Timestamp, then it will be converted to a datetime object.
    If the "dt" object is a string, then it will be parsed into a datetime object.
    """
    if isna(dt):
        if nan_to_now:
            return datetime.now()
        else:
            return None
    elif isinstance(dt, pd.Timestamp):  # pd.Timestamp will also isinstance True for datetime and date, so here we put it first
        return dt.to_pydatetime()
    elif isinstance(dt, datetime):
        return dt
    elif isinstance(dt, date):
        return datetime.combine(dt, datetime.min.time())
    else:
        if '+' in dt:
            try:
                return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S+00:00")
            except ValueError:
                pass
        init_split = dt.split(' ')
        if len(init_split) == 1:
            s = f'{dt} 00:00:00.000000'
        else:
            hour_min_sec_split = init_split[1].split(':')
            if len(hour_min_sec_split) == 1:
                s = f'{dt}:00:00.000000'
            elif len(hour_min_sec_split) == 2:
                s = f'{dt}:00.000000'
            else:
                if '.' not in hour_min_sec_split[2]:
                    s = f'{dt}.000000'
                else:
                    s = dt
        sep_chars = ['-', '/', '_', ',', '|', '~', ';']
        for sep in sep_chars:
            if sep in s:
                s = s.replace(sep, '-')
        combos = ['-'.join([f'%{j}' for j in i]) + ' %H:%M:%S.%f' for i in itertools.permutations('Ymd')]
        for strftime in combos:
            try:
                return datetime.strptime(s, strftime)
            except ValueError:
                pass
        else:
            return None


def debug_print_dict(
        dictionary: dict, 
        start_tabs: int = 0, 
        expand_dict_in_iterables: bool = True, 
        add_types: bool = False, 
        comma_numeric: bool = True,
        round_floats_to: Optional[int] = 3
    ) -> None:
    #######################################################################
    def expand_iterable(iterable: Union[tuple, list], tabs: int):
        tt = '\t' * tabs
        for item in iterable:
            if isinstance(item, dict):
                debug_print_dict(dictionary=item, start_tabs=tabs + 1)
            else:
                print(f'{tt}{item}')
    #######################################################################
    initial = start_tabs == 0
    if initial:
        # print('\n\n' + ('*' * 200))
        print('*' * 200)
    t = '\t' * start_tabs
    for k, v in dictionary.items():
        if isinstance(v, dict):
            if add_types:
                print(f'{t}{k}: {type(v)}')
            else:
                print(f'{t}{k}:')
            debug_print_dict(dictionary=v, start_tabs=start_tabs+1, expand_dict_in_iterables=expand_dict_in_iterables, add_types=add_types)
        elif isinstance(v, (tuple, list)) and expand_dict_in_iterables:
            if add_types:
                print(f'{t}{k}: {type(v)}')
            else:
                print(f'{t}{k}:')
            expand_iterable(iterable=v, tabs=start_tabs+1)
        else:
            val = v
            if isinstance(v, (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
                if isinstance(v, (float, np.float16, np.float32, np.float64)) and round_floats_to is not None:
                    val = round(v, round_floats_to)
                if comma_numeric:
                    val = f'{val:,}'

            if add_types:
                print(f'{t}{str(k) + ":":25}{val} {type(v)}')
            else:
                print(f'{t}{str(k) + ":":25}{val}')
    if initial:
        # print(('*' * 200) + '\n\n')
        print('*' * 200)


def debug_print_obj_dict(obj: object, start_tabs: int = 0, expand_dict_in_iterables: bool = True, add_types: bool = False) -> None:
    debug_print_dict(
        dictionary=obj.__dict__, 
        start_tabs=start_tabs, 
        expand_dict_in_iterables=expand_dict_in_iterables, 
        add_types=add_types
    )


def ensure_list(value: Any, return_none_if_none: bool = False) -> Optional[list]:
    if return_none_if_none and isna(value):
        return None

    if isinstance(value, list):
        return value
    elif isinstance(value, tuple):
        return list(value)
    else:
        return [value]
    

def filter_callable_kwargs(
        func: Callable,
        passed_kwargs: dict,
        remove_predefined_kwargs: Optional[list] = None,
        filter_func: Optional[Callable] = None,
        convert_numpy_values: bool = False,
) -> dict:
    """
    Filters the keyword arguments, getting only the key-value pairs that can
    actually be passed to a particular function/method ("func")

    :param func:                        A function/method to be inspected
    :param passed_kwargs:               The keyword arguments trying to be passed to the function/method
    :param remove_predefined_kwargs:    A list of keys within the "passed_kwargs" dict that should be removed, assuming
                                        they are already defined in actual function call
    :param filter_func:                 An optional function that can be passed to further filter the kwargs, this function should
                                        return either True or False based on the key of the passed_kwargs
    :return: dict - The filtered keyword arguments
    """
    filtered_kwargs = {}
    passed_kwargs = dict(**passed_kwargs)  # This is essentially to "copy" the kwargs, as the base object (such as a post) can change asynchronously

    # Functions in a decorator will have a __wrapped__ attribute that points to the original function
    # if they are wrapped using functools.wraps within the decorator, this is good practice when writing decorators
    # so that it preserves the original function's metadata
    # Here we check if the func has the __wrapped__ attribute, if it does, then use that to get the args,
    # otherwise decorated the inspect module can only find the args of the decorator function
    if hasattr(func, '__wrapped__'):
        args = inspect.getfullargspec(func.__wrapped__).args
    else:
        args = inspect.getfullargspec(func).args
    for k, v in passed_kwargs.items():
        if k in args:
            if filter_func is not None:
                if filter_func(k):
                    if convert_numpy_values:
                        filtered_kwargs[k] = convert_numpy_value(value=v)
                    else:
                        filtered_kwargs[k] = v
            else:
                if convert_numpy_values:
                    filtered_kwargs[k] = convert_numpy_value(value=v)
                else:
                    filtered_kwargs[k] = v
    if remove_predefined_kwargs:
        filtered_kwargs = remove_keys_from_dict(dictionary=filtered_kwargs, keys=remove_predefined_kwargs)
    return filtered_kwargs


def format_data(
        data: pd.DataFrame, 
        inplace: bool = False, 
        column_mappers: Optional[dict] = None,
        clean_columns: bool = True,
        buy_column: Optional[str] = None,
        sell_column: Optional[str] = None
    ) -> pd.DataFrame:
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if inplace:
            use_data = data
        else:
            use_data = data.copy(deep=True)
        required_count = 0
        rename_cols = {}
        drop_cols = []

        if use_data.index.name == 'date':
            date_is_index = True
            required_cols.remove('date')
        else:
            date_is_index = False

        # Do a column check
        for col in use_data.columns:
            c_lower = col.lower()
            if c_lower in required_cols:
                rename_cols[col] = c_lower
                required_count += 1
            elif column_mappers is not None and col in column_mappers and column_mappers[col] in required_cols:
                required_count += 1
                rename_cols[col] = column_mappers[col]
            else:
                drop_cols.append(col)

        # Rename columns if applicable 
        if rename_cols:
            use_data.rename(columns=rename_cols, inplace=True)
        
        # Drop columns if applicable
        if clean_columns and drop_cols:
            use_data.drop(columns=drop_cols, inplace=True)

        # Check if all required columns are present
        if required_count != len(required_cols):
            cols_not_found = [col for col in required_cols if col not in use_data.columns]
            raise ValueError(f'Required columns: {cols_not_found} not found in data')
        
        # Set the index to the date column and sort it
        if not date_is_index:
            use_data.set_index('date', inplace=True)
        use_data.sort_index(inplace=True)

        # Add the buy and sell columns and initialize them to np.nan, if applicable
        if buy_column is not None:
            use_data[buy_column] = np.nan
        if sell_column is not None:
            use_data[sell_column] = np.nan
        return use_data


def generate_alnum_id(id_length: int = 8) -> str:
    """
    Generates a random alphanumeric id
    :param id_length:   The length of the id
    :return:        str - The id
    """
    nums = [48, 57]
    chrs = [65, 90]
    select = [nums, chrs]
    alnum_id = ''
    for i in range(id_length):
        choice = random.choice(select)
        alnum_id += chr(random.randint(*choice))
    return alnum_id


def generate_uuid() -> str:
    """
    Generates a universal unique identifier (UUID) version 4
    :return: str - uuid4
    """
    return str(uuid4())


def get_default_kwargs(func: Callable) -> dict:
    args = inspect.getfullargspec(func)
    positional_count = len(args.args) - len(args.defaults)
    defaults = dict(zip(args.args[positional_count:], args.defaults))
    return defaults


def get_dt_at_n_periods(dt: datetime, interval: T.Trade.INTERVAL, n_periods: int) -> datetime:
    if interval.endswith('m'):
        change = int(interval[:-1])
        new_dt = dt + timedelta(minutes=change * n_periods)
        minus_td = timedelta(minutes=change)
    elif interval.endswith('h'):
        change = int(interval[:-1])
        new_dt = dt + timedelta(hours=change * n_periods)
        minus_td = timedelta(hours=change)
    elif interval.endswith('d'):
        change = int(interval[:-1])
        new_dt = dt + timedelta(days=change * n_periods)
        minus_td = timedelta(days=change)
    elif interval.endswith('wk'):
        change = int(interval[:-2])
        new_dt = dt + timedelta(weeks=change * n_periods)
        minus_td = timedelta(weeks=change)
    elif interval.endswith('mo'):
        change = int(interval[:-2])
        new_dt = dt + timedelta(days=change * 30 * n_periods)
        minus_td = timedelta(days=change * 30)
    else:  # endswith('yr')
        change = int(interval[:-2])
        new_dt = dt + timedelta(days=change * 365 * n_periods)
        minus_td = timedelta(days=change * 365)
    next_dt = get_next_interval_dt(interval=interval, dt=new_dt)
    return next_dt - minus_td


def get_fibonaaci_levels(max_price: float, min_price: float) -> dict:
    diff = max_price - min_price
    fib_levels = {
        '0.00': max_price,
        '0.236': max_price - (diff * 0.236),
        '0.382': max_price - (diff * 0.382),
        '0.500': max_price - (diff * 0.500),
        '0.618': max_price - (diff * 0.618),
        '0.786': max_price - (diff * 0.786),
        '1.00': min_price
    }
    return fib_levels


def get_next_interval_dt(interval: T.Trade.INTERVAL, dt: Optional[datetime] = None) -> datetime:
    # Convert to UTC if necessary
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt is None:
        dt = datetime.now(tz=timezone.utc)

    if interval.endswith('m'):
        change = int(interval[:-1])
        next_gross_dt = dt + timedelta(minutes=change)
        next_interval_minutes = next_gross_dt.minute - (next_gross_dt.minute % change)
        next_dt = datetime(
            year=next_gross_dt.year,
            month=next_gross_dt.month,
            day=next_gross_dt.day,
            hour=next_gross_dt.hour,
            minute=next_interval_minutes,
            tzinfo=timezone.utc
        )
    elif interval.endswith('h'):
        change = int(interval[:-1])
        next_gross_dt = dt + timedelta(hours=change)
        next_interval_hours = next_gross_dt.hour - (next_gross_dt.hour % change)
        next_dt = datetime(
            year=next_gross_dt.year,
            month=next_gross_dt.month,
            day=next_gross_dt.day,
            hour=next_interval_hours,
            tzinfo=timezone.utc
        )
    elif interval.endswith('d'):
        change = int(interval[:-1])
        next_dt = dt + timedelta(days=change)
        next_dt = datetime(
            year=next_dt.year,
            month=next_dt.month,
            day=next_dt.day,
            tzinfo=timezone.utc
        )
    elif interval.endswith('wk'):
        change = int(interval[:-2])
        next_dt = dt + timedelta(weeks=change)
        next_dt = datetime(
            year=next_dt.year,
            month=next_dt.month,
            day=next_dt.day,
            tzinfo=timezone.utc
        )
    elif interval.endswith('mo'):
        change = int(interval[:-2])
        next_dt = dt + timedelta(days=change)
        next_dt = datetime(
            year=next_dt.year,
            month=next_dt.month,
            day=next_dt.day,
            tzinfo=timezone.utc
        )
        
    else:  # endswith('yr')
        change = int(interval[:-2])
        next_dt = dt + timedelta(days=365 * change)
        next_dt = datetime(
            year=next_dt.year,
            month=next_dt.month,
            day=next_dt.day,
            tzinfo=timezone.utc
        )
    return next_dt


def stddevs_by_period_length(self, data: pd.DataFrame, target_column = 'close', period_length: int = 120) -> tuple[list, list]:
    upper_stds = []
    lower_stds = []
    #############################################################
    def update_standard_devs(df_chunk):
        mean = df_chunk['close'].mean()
        std = df_chunk['close'].std()
        upper_stds.extend([mean + std] * len(df_chunk))
        lower_stds.extend([mean - std] * len(df_chunk))
    #############################################################
    len_data = len(data)
    data['idx'] = range(len_data)
    if len_data < period_length:
        return [np.nan] * len_data, [np.nan] * len_data

    # Get the standard deviations to use for each chunk
    previous_idx = -1
    at_idx = period_length

    while at_idx < len_data:
        df_chunk = data.loc[((previous_idx < data['idx']) & (data['idx'] <= at_idx))]
        update_standard_devs(df_chunk)
        previous_idx = at_idx
        at_idx += period_length
    df_chunk = data.loc[((previous_idx < data['idx']) & (data['idx'] <= len_data))]
    update_standard_devs(df_chunk)
    data.drop(columns=['idx'], inplace=True)
    return upper_stds, lower_stds
    

def isna(value: Any, empty_str_isna: bool = False, zero_isna: bool = False) -> bool:
    # Pandas isn't always consistent about None or NaN values for null values
    # None and NaN are not equivalent, for example "if value is None" will return False for NaN values
    # so here we see if the value is a catch-all null value
    it_isna = False
    if value is None or value is np.nan:
        it_isna = True
    elif empty_str_isna and value == '':
        it_isna = True
    elif zero_isna and value == 0:
        it_isna = True

    if it_isna:
        return it_isna

    try:
        if pd.isna(value):
            it_isna = True
    except ValueError:
        # Pandas isna can get tripped up with iterable types
        # so it is safe to assume that if an exception was raised
        # there was in fact a non-null value that threw a ValueError
        pass
    return it_isna


def isna_all(*values: Any, empty_str_isna: bool = False, zero_isna: bool = False) -> bool:
    # Returns True is all values within the values are None or pd.isna
    return len(values) == sum([isna(value, empty_str_isna=empty_str_isna, zero_isna=zero_isna) for value in values])


def isna_any(*values: Any, empty_str_isna: bool = False, zero_isna: bool = False) -> bool:
    # Return True is any value within values is None or pd.isna
    return sum([isna(value, empty_str_isna=empty_str_isna, zero_isna=zero_isna) for value in values]) > 0


def monte_carlo(
        data: pd.DataFrame,
        interval: T.Trade.INTERVAL,
        n_periods: int,
        n_runs: int,
        use_data_inplace: bool = False,
        data_column_mappers: Optional[dict] = None,
        mc_type: T.Strategy.MONTE_CARLO = 'normal',
        show_plot: bool = False,
        show_historical: bool = False,
        keep_historical: bool = True,
        price_min_bound: Optional[float] = None,
        price_max_bound: Optional[float] = None,
        induce_volatility: bool = False,
        volatility_max_std: float = 0.075
    ) -> pd.DataFrame:
    if price_min_bound is None or price_min_bound < 0:
        price_min_bound = 0
    ########################################################################################
    def bound_future_prices(future_prices: np.array) -> np.array:
        if price_min_bound is not None:
            future_prices[future_prices < price_min_bound] = price_min_bound
        if price_max_bound is not None:
            future_prices[future_prices > price_max_bound] = price_max_bound
        return future_prices
    
    def format_title(final_price: float, final_datetime: datetime, start_datetime: datetime, round_to: int = 2) -> str:
        title = f'Monte Carlo Simulations ({mc_type.capitalize()})\n'
        title += f'{n_runs} Runs, {interval} Intervals, Start {start_datetime.strftime("%m-%d-%Y %H:%M")}\n'
        title += f'Final Price: ${final_price:,.{round_to}f} on {final_datetime.strftime("%m-%d-%Y %H:%M")}'
        return title
    
    def get_normal_distribution_and_future_prices(final_close_price: float):
        closes_normal = np.random.normal(
            loc=mean, 
            scale=std, 
            size=n_periods
        )
        cum_closes = np.random.choice(closes_normal, n_periods, replace=True).cumsum()
        future_prices = final_close_price * (1 + cum_closes)
        return bound_future_prices(future_prices=future_prices)
    ######################################################################################## 
    # Copy and format the data
    df = format_data(data=data, inplace=use_data_inplace, column_mappers=data_column_mappers)

    # Get the percentage change of the close price
    close_price_pct_changes = df['close'].pct_change().dropna()
    high_diff_percentage = (df['high'] - df['close']) / df['close']
    low_diff_percentage = (df['low'] - df['close']) / df['close']

    # Initialize simulations matrix by the number of runs and periods (ahead) 
    simulations = np.zeros((n_runs, n_periods))

    if mc_type == 'direct':
        # Run the simulations and randomly select the percentage change to apply to each period
        # The "direct" terms means that the future prices is directly related to random actual percentage changes
        for n_run in range(n_runs):
            cum_closes = np.random.choice(close_price_pct_changes, n_periods, replace=True).cumsum()
            future_prices = df['close'].iloc[-1] * (1 + cum_closes)
            simulations[n_run, :] = bound_future_prices(future_prices=future_prices)
    
    elif mc_type == 'brownian':
        # Get the mean and standard deviation of the percentage changes
        mean = close_price_pct_changes.mean()
        std = close_price_pct_changes.std()

        # Calculate the drift of the percentage changes
        drift = mean * std ** 2 / 2

        # Run the simulations and use the goemetric brownian motion algorithm to get the future prices
        for n_run in range(n_runs):
            # Calculate the diffusion using a random normal distibution cumulative sum multiplied by the 
            # standard deviation to get random future prices for each period
            normal_cumsum = np.random.normal(loc=0, scale=1, size=n_periods).cumsum()
            diffusion = std * normal_cumsum
            future_prices = df['close'].iloc[-1] * np.exp(drift + diffusion)
            simulations[n_run, :] = bound_future_prices(future_prices=future_prices)
    
    # Include more options for the monte carlo simulations
    
    else:  # mc_type == 'normal'
        # Get the mean and standard deviation of the percentage changes
        mean = close_price_pct_changes.mean()
        std = close_price_pct_changes.std()

        # Use the get_normal_distribution_and_future_prices() function, which will
        # get a new normal distibution for every run and then get the future
        # prices for each period by randomly selecting a percentage change from the
        # distribution for each period     

        # Run the simulations and get a new percentage change disribution
        # for each run for which to randomly select from for each period
        for n_run in range(n_runs):
            simulations[n_run, :] = get_normal_distribution_and_future_prices(final_close_price=df['close'].iloc[-1])
    
    # If induce_volatility is True, randomly select a simulatioon where the final price is within 
    # "volatility_max_std" standard deviations of the mean final price
    # TODO: Figue out a way to get the average before bounding the future prices, to get the true trend but the bound
    # the prices after
    if induce_volatility:
        final_price_mean = simulations[:, -1].mean()
        final_price_std = simulations[:, -1].std()
        valid_sims = simulations[(
            (simulations[:, -1] > final_price_mean - (volatility_max_std * final_price_std)) 
            & (simulations[:, -1] < final_price_mean + (volatility_max_std * final_price_std))
        )]
        avg_prices = random.choice(valid_sims)
    else:
        # Get the average price for each period ahead
        avg_prices = simulations.mean(axis=0)

    # Format the average prices to put into the historical data DataFrame
    # monte_carlo_data = {
    #     'date': [],
    #     'montecarlo': [],
    #     'close': [],
    #     'open': [np.nan] * n_periods,
    #     'high': [np.nan] * n_periods,
    #     'low': [np.nan] * n_periods,
    #     'volume': [np.nan] * n_periods
    # }
    monte_carlo_data = {
        'date': [],
        'montecarlo': [],
        'close': [],
        'open': [np.nan] * n_periods,
        'high': [],
        'low': [],
        'volume': [np.nan] * n_periods
    }
    last_interval_dt = df.index[-1]
    for n_period in range(n_periods):
        # Calculate the next interval datetime
        last_interval_dt = get_next_interval_dt(interval, last_interval_dt)
        monte_carlo_data['date'].append(last_interval_dt)
        monte_carlo_data['montecarlo'].append(True)
        avg_close = avg_prices[n_period]
        monte_carlo_data['close'].append(avg_close)
        # Get the high and low prices by randomly selecting from the high and low diff percentages
        monte_carlo_data['high'].append(avg_close * random.choice(high_diff_percentage))
        monte_carlo_data['low'].append(avg_close * random.choice(low_diff_percentage))
    
    monte_carlo_df = pd.DataFrame(monte_carlo_data)
    monte_carlo_df.set_index('date', drop=True, inplace=True)

    if show_plot:
        monte_prophet_show_sims(
            preditions=monte_carlo_df,
            historical=df,
            title=format_title(
                final_price=monte_carlo_df['close'].iloc[-1],
                final_datetime=monte_carlo_df.index[-1],
                start_datetime=df.index[0],
                round_to=currency_round_to(max_price=monte_carlo_df['close'].max(), min_price=monte_carlo_df['close'].min())
            ),
            simulations=simulations,
            show_historical=show_historical
        )
    if keep_historical:
        df['montecarlo'] = False
        df = pd.concat([df, monte_carlo_df], axis=0)
        return df
    else:
        monte_carlo_df.drop(columns=['montecarlo'], inplace=True)
        return monte_carlo_df


def currency_round_to(max_price: float, min_price: float) -> int:
    if -1 < min_price < 1 or -1 < max_price < 1:
        return 4
    return 2


def monte_prophet_show_sims(
        preditions: pd.DataFrame, 
        historical: pd.DataFrame, 
        title: str,
        simulations: Optional[np.array] = None,
        show_historical: bool = False
    ) -> None:
    with TimerInline('Plotting Monte Carlo Simulations'):
        round_to = 2
        min_price = preditions['close'].min()
        max_price = preditions['close'].max()
        if min_price < 1 or max_price < 1:
            round_to = 4 
        fig, ax = plt.subplots(figsize=(15, 8))
        if show_historical:
            ax.plot(historical.index, historical['close'], color='blue', linewidth=2)
        if simulations is not None:
            max_sims = simulations.max(axis=0)
            min_sims = simulations.min(axis=0)
            ax.fill_between(preditions.index, min_sims, max_sims, color='black', alpha=0.3)
        ax.plot(preditions.index, preditions['close'], color='red', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted Close Price ($)')
        ax.set_title(title)
        ax.yaxis.set_major_formatter(StrMethodFormatter(f'${{x:,.{round_to}f}}'))
        ax.xaxis.set_major_formatter(DateFormatter(f'%m-%d-%Y %H:%M'))                
    plt.show()


def prophet_predict(
        data: pd.DataFrame,
        interval: T.Trade.INTERVAL,
        n_periods: int,
        data_column_mappers: Optional[dict] = None,
        return_price: Literal['average', 'low', 'high', 'random'] = 'average',
        price_min_bound: Optional[float] = None,
        price_max_bound: Optional[float] = None,
        show_plot: bool = False,
        show_historical: bool = False,       
        keep_historical: bool = True 
    ) -> pd.DataFrame:
    if price_min_bound is None or price_min_bound < 0:
        price_min_bound = 0
    # Copy and format the data
    df = format_data(data=data, inplace=False, column_mappers=data_column_mappers)
    df_formatted = pd.DataFrame({
        'ds': df.index,
        'y': df['close']
    })
    # Remove timezone if it exists but store it for later
    tz_info = None
    if df_formatted['ds'].dt.tz is not None:
        tz_info = (df_formatted['ds'].dt.tz, )  # Store as tuple for immutability
        df_formatted['ds'] = df_formatted['ds'].dt.tz_localize(None)
        df_formatted['ds'] = pd.to_datetime(df_formatted['ds'])

    # Fit the model and predict the future
    pro = Prophet()
    pro.fit(df_formatted)
    future = pro.make_future_dataframe(
        periods=n_periods, 
        freq=pd_freq_from_interval(interval)
    )
    forecast = pro.predict(future)
    # Return timzone if it existed
    if tz_info is not None:
        forecast['ds'] = forecast['ds'].dt.tz_localize(tz_info[0])
    forecast = forecast.loc[forecast['ds'] > df.index[-1]]
    for y_name in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast.loc[forecast[y_name] < price_min_bound, y_name] = price_min_bound
        if price_max_bound is not None:
            forecast.loc[forecast[y_name] > price_max_bound, y_name] = price_max_bound
    
    len_forecast = len(forecast)
    predict_data = {
        'date': forecast['ds'],
        'close': [],
        'prophet': [True] * len_forecast
    }
    if return_price == 'average':
        predict_data['close'] = forecast['yhat']
    elif return_price == 'low':
        predict_data['close'] = forecast['yhat_lower']
    elif return_price == 'high':
        predict_data['close'] = forecast['yhat_upper']
    else:  # return_price == 'random'
        for idx in forecast.index:
            choices = [forecast['yhat'].iloc[idx], forecast['yhat_lower'].iloc[idx], forecast['yhat_upper'].iloc[idx]]
            predict_data['close'].append(random.choice(choices))
    for col in df.columns:
        if col not in predict_data:
            predict_data[col] = [np.nan] * len_forecast
    predict_df = pd.DataFrame(predict_data)
    predict_df.set_index('date', drop=True, inplace=True)

    if show_plot:        
        monte_prophet_show_sims(
            preditions=predict_df,
            historical=df,
            title='Facebook Prophet Predictions',
            simulations=None,
            show_historical=show_historical
        )
    if keep_historical:
        df['prophet'] = False
        return pd.concat([df, predict_df], axis=0)
    else:
        predict_df.drop(columns=['prophet'], inplace=True)
        return predict_df

    

    
def n_periods_from_timedelta(td: timedelta, interval: T.Trade.INTERVAL) -> int:
    if interval.endswith('m'):
        change = int(interval[:-1])
        return int(td.total_seconds() / 60 / change)
    elif interval.endswith('h'):
        change = int(interval[:-1])
        return int(td.total_seconds() / 3600 / change)
    elif interval.endswith('d'):
        change = int(interval[:-1])
        return int(td.total_seconds() / 86_400 / change)
    elif interval.endswith('wk'):
        change = int(interval[:-2])
        return int(td.total_seconds() / (86_400 * 7) / change)
    elif interval.endswith('mo'):
        change = int(interval[:-2])
        return int(td.total_seconds() / (86_400 * 30) / change)
    else:  # endswith('yr')
        change = int(interval[:-2])
        return int(td.total_seconds() / (86_400 * 265) / change)


def nan() -> float:
    return float('nan') 


def pd_freq_from_interval(interval: T.Trade.INTERVAL) -> str:
    if interval.endswith('m'):
        return interval + 'in' # '5m -> 5min'
    elif interval.endswith('h'):
        return interval
    elif interval.endswith('d'):
        return interval.upper()
    elif interval.endswith('wk'):
        change = int(interval[:-2])
        return f'{change}W'
    elif interval.endswith('mo'):
        change = int(interval[:-2])
        return f'{change}M'
    else:  # endswith('yr')
        change = int(interval[:-2])
        return f'{change}Y' 


def prepare_stringify_value(value: Any, none_is_null: bool = True, bool_is_int: bool = True, quote: Literal["'", '"'] = "'") -> str:
    """
    Generate a Stringified List of values converting them from Python
    Quotes within strings are removed

        str -> "a string" || 'a string'
        int -> str(int)
        float -> str(float)
        bool -> True || False || 1 || 0
        None -> NULL || None

    Example
        iterable = ['a string', 42, 43.5, True, None, "a string with ' apostrophe"]
        return -> "('a_string', 42, 43.5, 1, NULL, 'a string with apostrophe')"
    :param iterable: A sequences of values such as a list or tuple
    :return: str
    """
    if value is None:
        if none_is_null:
            return 'NULL'
        else:
            return 'None'
    elif isinstance(value, str):
        value = value.replace('"', '').replace("'", '')
        return f"{quote}{value}{quote}"
    elif isinstance(value, bool):
        if bool_is_int:
            return str(int(value))
        else:
            return str(value)
    elif isinstance(value, (date, datetime, pd.Timestamp)):
        return f"DATETIME({quote}{value.strftime('%Y-%m-%d %H:%M:%S')}{quote})"
    else:
        return str(value)

    
def pretty_time(seconds: float, round_to_small: int = 6, round_to_big: int = 1) -> str:
        """
        This returns the seconds in a pretty format, either as seconds,
        minutes, hours, or days, depending on the size of the input.
        """
        if seconds < 1:
            return f'{round(seconds, round_to_small)} sec'
        elif seconds < 60:
            if seconds == 1:
                return f'{round(seconds, round_to_small)} sec'
            else:
                return f'{round(seconds, round_to_big)} secs'
        elif seconds < 3_600:
            minutes = int(seconds // 60)
            if minutes == 1:
                return f'{minutes} min, {pretty_time(seconds - (minutes * 60), round_to_small=round_to_small, round_to_big=round_to_big)}'
            else:
                return f'{minutes} mins, {pretty_time(seconds - (minutes * 60), round_to_small=round_to_small, round_to_big=round_to_big)}'
        elif seconds < 86_400:
            hours = int(seconds // 3_600)
            if hours == 1:
                return f'{hours} hour, {pretty_time(seconds - (hours * 3_600), round_to_small=round_to_small, round_to_big=round_to_big)}'
            else:
                return f'{hours} hours, {pretty_time(seconds - (hours * 3_600), round_to_small=round_to_small, round_to_big=round_to_big)}'
        else:
            days = int(seconds // 86_400)
            if days == 1:
                return f'{days} day, {pretty_time(seconds - (days * 86_400), round_to_small=round_to_small, round_to_big=round_to_big)}'
            else:
                return f'{days} days, {pretty_time(seconds - (days * 86_400), round_to_small=round_to_small, round_to_big=round_to_big)}'


def prtcolor(text: str, color_code: int, prefix: Optional[str] = None, add_prefix: bool = True, add_newline: bool = True) -> None:
    """
    Prints a colored message to the console.
    :param text: The message to display
    :param color_code: The color code for the formatting
        black      30
        red        31
        green      32
        yellow     33
        blue       34
        magenta    35
        cyan       36
        white      37
    :param prefix: A prefix to the message such as 'WARNING' or 'INFO'
    :param add_prefix: Boolean indicating if the prefix should be added, if prefix is None, this can be ignored
    :param add_newline: Boolean indicating if a newline should be added after the message
    :return:    None - prints the message to the console
    """
    if prefix is not None and add_prefix:
        prefix += ':\t'
        print_statement = f'\033[0;{color_code};49m{prefix}{text} \033[0m'
    else:
        print_statement = f'\033[0;{color_code};49m{text} \033[0m'
    if add_newline:
        print_statement += '\n'
    print(print_statement)


def prtinfo(text: str, add_newline: bool = True) -> None:
    prtcolor(text=text, color_code=34, prefix='[INFO]', add_prefix=True, add_newline=add_newline)


def prtwarn(text: str, add_newline: bool = True) -> None:
    prtcolor(text=text, color_code=33, prefix='[WARNING]', add_prefix=True, add_newline=add_newline)


def ordered_set(iterable: Sequence) -> list:
    """
    Returns a list of unique values while maintaining the order of the original iterable
    :param iterable:    A sequence of values such as a list or tuple
    :return:    list
    """
    list_set = []
    for value in iterable:
        if value not in list_set:
            list_set.append(value)
    return list_set


def remove_keys_from_dict(dictionary: dict, keys: list) -> dict:
    """
    Removes keys from a dictionary
    :param dictionary:  The dictionary to remove keys from
    :param keys:        A list of keys to remove
    :return:    dict
    """
    for key in keys:
        dictionary.pop(key, None)
    return dictionary


def send_email(
        to: Union[str, list[str]], 
        subject: str,
        message: str,
        message_contains_html: bool = False
    ) -> None:    
    ###################################################################################
    def _format_to(to_from_args) -> list:
        if isinstance(to_from_args, str):
            if ',' in to_from_args:
                to_list = [i.replace(' ', '') for i in to_from_args.split(',')]
            elif ';' in to_from_args:
                to_list = [i.replace(' ', '') for i in to_from_args.split(';')]
            else:
                to_list = [to_from_args]
        else:
            to_list = [str(i) for i in to_from_args]
        return to_list
    ###################################################################################
    to = _format_to(to)
    print(f'{to=}')
    try:
        from_addr = EnvReader.get('MAIL_USERNAME')
        msg = MIMEMultipart('alternative')
        msg['From'] = from_addr
        msg['To'] = ', '.join(to)
        msg['Subject'] = subject
        if message_contains_html:
            msg.attach(MIMEText(message, 'html'))
        else:
            msg.attach(MIMEText(message, 'plain'))

        smtp = smtplib.SMTP_SSL(
            host=EnvReader.get('MAIL_SERVER'),
            port=EnvReader.get('MAIL_PORT'),
        )
        smtp.ehlo()
        smtp.login(user=from_addr, password=EnvReader.get('MAIL_PASSWORD'))
        smtp.sendmail(from_addr=from_addr, to_addrs=to, msg=msg.as_string())
        smtp.close()

        print(f"Email successfully sent to {', '.join(to)}")
    except Exception as e:
        print(f'Could not send email TB:\n{traceback.format_exc()}')


def sqlize_list(iterable: Sequence, return_as_list: bool = False) -> Union[str, list]:
    sql_values = [prepare_stringify_value(value=value, none_is_null=True, bool_is_int=True, quote="'") for value in iterable]
    if return_as_list:
        return sql_values
    else:
        return f"({', '.join(sql_values)})"


def sql_in_statement_check_length(iterable: Sequence, variable_name: str, max_length: int = 999) -> str:    
    values = ensure_list(iterable)
    if len(values) <= max_length:
        return f'{variable_name} IN {sqlize_list(values)}'
    else:
        sql = '('
        in_chunks = []
        for value_chunk in chunker(values, 999):
            in_chunks.append(f"{variable_name} IN {sqlize_list(value_chunk)}")
        sql += ' OR '.join(in_chunks) + ')'
        return sql


def startswith(value: str, startswith_check: Union[str, Sequence[str]]) -> bool:
    """
    Returns True if the value starts with any of the startswith_check values
    :param value:               The value to check
    :param startswith_check:    A string or list of strings to check against
    :return:    bool
    """
    if isinstance(startswith_check, str):
        return value.startswith(startswith_check)
    else:
        return sum([value.startswith(s) for s in startswith_check]) > 0


def stddevs_by_period_length(
        data: pd.Series, 
        period_length: int = 120, 
        std_multiplier: float = 1, 
        trending_on: bool = True,
        trend_divisor: float = 5,
    ) -> tuple[list, list]:
    upper_stds = []
    lower_stds = []
    #############################################################
    def update_standard_devs(previous_chunk: pd.Series, current_chunk: pd.Series, previous_mean: float, trend_count: int):
        mean = previous_chunk.mean()
        std = previous_chunk.std()
        if trending_on:
            if previous_mean is None:
                previous_mean = mean
            else:
                if mean > previous_mean:
                    trend_count += 1
                elif mean < previous_mean:
                    trend_count -= 1
                previous_mean = mean
            mean += ((trend_count / trend_divisor) * std)
        upper_stds.extend([mean + (std * std_multiplier)] * len(current_chunk))
        lower_stds.extend([mean - (std * std_multiplier)] * len(current_chunk))
        return previous_mean, trend_count
    #############################################################
    len_data = len(data)
    if len_data < period_length:
        return [np.nan] * len_data, [np.nan] * len_data

    temp_df = pd.DataFrame({
        'target': data,
        'idx': range(len_data)
    })
    trend_count = 0
    previous_mean = None
    int_period_length = int(period_length)
    upper_stds.extend([np.nan] * int_period_length)
    lower_stds.extend([np.nan] * int_period_length)
    chunks = list(chunker(range(len_data), int_period_length))
    for i, chunk in enumerate(chunks[1:], 1):
        previous_chunk = temp_df.loc[((chunks[i-1][0] <= temp_df['idx']) & (temp_df['idx'] <= chunks[i-1][-1])), 'target']
        current_chunk = temp_df.loc[((chunk[0] <= temp_df['idx']) & (temp_df['idx'] <= chunk[-1])), 'target']
        previous_mean, trend_count = update_standard_devs(previous_chunk, current_chunk, previous_mean, trend_count)
    return upper_stds, lower_stds


def tablefy_dict(dictionary: dict, max_display_length: int = 17) -> str:
    """
    Returns a formatted string of a dictionary in a table format
    :param dictionary:          The dictionary to format
    :param max_key_length:      The maximum length of the keys
    :param max_value_length:    The maximum length of the values
    :return:    str
    """
    headers = []
    values = []
    for key, value in dictionary.items():
        key_str = str(key)
        value_str = str(value)
        if len(key_str) > max_display_length:
            key_str = key_str[:max_display_length - 3] + '...'
        if len(value_str) > max_display_length:
            value_str = value_str[:max_display_length - 3] + '...'
        headers.append(key_str)
        values.append(value_str)
    headers = [f'{h:<{max_display_length}}' if i == 0 else f'{h:^{max_display_length}}' for i, h in enumerate(headers)]
    values = [f'{v:<{max_display_length}}' if i == 0 else f'{v:^{max_display_length}}' for i, v in enumerate(values)]
    breaks = ''.join([('-' * max_display_length) + '|' for _ in range(len(dictionary))]) + '\n'
    table = breaks + f"{'|'.join(headers)}|\n" + breaks + f"{'|'.join(values)}|\n" + breaks[:-1]
    return table


def tablefy_dict_html(dictionary: dict) -> str:
    """
    Returns a formatted string of a dictionary in a table format
    :param dictionary:          The dictionary to format
    :param max_key_length:      The maximum length of the keys
    :param max_value_length:    The maximum length of the values
    :return:    str
    """
    html = """
    <html>
        <head>
            <style>
                table, td {{
                    border: 1px solid black;
                    border-collapse: collapse;
                    padding: 5px;
                }}
            </style>
        </head>
        <body>
            {table_block}
        </body>
    </html>
    """
    table_block = """
            <table>
                <tbody>
    """
    for key, value in dictionary.items():
        table_block += f'\t\t\t<tr><td>{key}</td><td>{value}</td></tr>\n'
    table_block += """
                </tbody>
            </table>
    """
    return html.format(table_block=table_block)


def ticker_check_pair(ticker: str, pair: str = 'USD', sep: str = '-', lower: bool = False) -> str:
    cryptos = ['BTC', 'DOGE', 'ETH', 'LTC', 'XRP', 'ADA', 'BCH', 'BNB', 'EOS', 'ETC', 'LINK', 'TRX', 'XLM', 'XTZ', 'XMR', 'ZEC']
    t_upper = ticker.upper()
    if t_upper in cryptos:
        with_pair = f'{t_upper}{sep}{pair}'
        if lower:
            return with_pair.lower()
        else:
            return with_pair.upper()
    else:
        if lower:
            return t_upper.lower()
        else:
            return t_upper


def timer(func: Optional[Callable] = None, print_starting: bool = True, prettify_time: bool = True, round_to: int=8) -> Callable:
    def run(ff, *a, **kw):
        if print_starting:
            prtcolor(text=f'Starting "{ff.__name__}"...', color_code=35, prefix='[TIMER WRAPPER]', add_prefix=True)
        start = time.time()
        result = ff(*a, **kw)
        elapsed = time.time() - start
        if prettify_time:
            text = f'Function: "{ff.__name__}" took {pretty_time(seconds=elapsed, round_to_small=round_to)} to complete.'
        else:
            text = f'Function: "{ff.__name__}" took {elapsed:,.{round_to}f} seconds to complete.'
        prtcolor(text=text, color_code=35, prefix='[TIMER WRAPPER]', add_prefix=True)
        return result

    if callable(func):
        # The decorator has been assigned without keyword args: @timer
        def wrapper(*args, **kwargs):
            return run(func, *args, **kwargs)
        return wrapper
    else:
        # The decorator has been assigned with keyword args: @timer(...)
        def decorator(f):
            def wrapper(*args, **kwargs):
                return run(f, *args, **kwargs)
            return wrapper
        return decorator


def timer_try_remove_file(filepath: str, try_attempts: int = 5, timer_seconds: int = 2) -> None:
    timer = Timer(timer_seconds, try_remove_file, kwargs={'filepath': filepath, 'try_attempts': try_attempts})
    timer.start()


def try_remove_file(filepath: str, try_attempts: int = 5) -> bool:
    removed = False
    for i in range(try_attempts):
        try:
            os.remove(filepath)
            removed = True
            break
        except Exception as e:
            time.sleep(0.25)
            pass
    return removed


def timer_try_recursive_remove_directory(directory: str, try_attempts: int = 5, timer_seconds: int = 2) -> None:
    timer = Timer(timer_seconds, try_recursive_remove_directory, kwargs={'directory': directory, 'try_attempts': try_attempts})
    timer.start()


def try_recursive_remove_directory(directory: str, try_attempts: int = 5) -> bool:
    """
    Recursively deletes all files and subdirectories within a directory, used for removing extracted
    field data folder from the inbox
    """
    removed = False
    for _ in range(try_attempts):
        try:
            for file in os.listdir(directory):
                path = os.path.join(directory, file)
                if os.path.isdir(path):
                    try_recursive_remove_directory(path)
                else:
                    os.remove(path)
            os.rmdir(directory)
            removed = True
            break
        except Exception as e:
            time.sleep(0.25)
            pass
    return removed


def unique_file_name(filename: str, extension: str, directory: str = os.getcwd()) -> str:
    filepath = os.path.join(directory, f'{filename}.{extension}')
    suffix = 1
    while os.path.isfile(filepath):
        filepath = os.path.join(directory, f'{filename}_{suffix}.{extension}')
        suffix += 1
    return filepath 

def utc_to_pst(utc_dt: datetime) -> datetime:
    return utc_dt - timedelta(hours=8)


# Utility Classes ####################################################################################################################################
class FromEnv(object):
    def __init__(self, env_filepath: str):
        if not os.path.isfile(env_filepath):
            raise FileNotFoundError(f'Could not locate ".env" file with path: "{env_filepath}"')
        self.env_filepath = env_filepath

    def get(self, key: str):
        key_pattern = re.compile(fr'({key})\ ?\=\ ?(.+)')
        with open(self.env_filepath, mode='r') as f:
            contents = f.read()
        matches = list(key_pattern.finditer(contents))
        if not matches:
            raise KeyError(f'Could not locate key "{key}" within .env file')
        return eval(matches[0].groups()[1])

EnvReader = FromEnv(env_filepath=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))


class TemporaryDirectory(object):
    THIS_DIR = os.path.dirname(__file__)
    NOT_DELETED = os.path.join(THIS_DIR, 'temp_directories_not_deleted.txt')

    def __init__(self, parent_directory: str = THIS_DIR, debug_keep_directory: bool = False):
        if not os.path.isdir(parent_directory):
            raise FileNotFoundError(f'Parent directory: "{parent_directory}" does not seem to exist')
        self.parent_directory = parent_directory
        self.directory = os.path.join(parent_directory, generate_alnum_id())
        self.debug_keep_directory = debug_keep_directory
        while os.path.isdir(self.directory):
            self.directory = os.path.join(parent_directory, generate_alnum_id())
        os.makedirs(self.directory)

    @classmethod
    def _try_remove_not_deleted(cls):
        if os.path.isfile(cls.NOT_DELETED):
            with open(cls.NOT_DELETED, mode='r') as f:
                contents = [i.replace('\n', '') for i in f.readlines()]
            for directory in contents:
                if os.path.isdir(directory):
                    cls.try_delete_dir(directory=directory)

    @classmethod
    def try_delete_dir(cls, directory: str, trys: int = 5) -> bool:
        for _ in range(trys):
            try:
                shutil.rmtree(path=directory)
                return True
            except:
                pass
        with open(cls.NOT_DELETED, 'a') as f:
            f.write(directory + '\n')
        return False

    def __enter__(self) -> str:
        return self.directory

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.debug_keep_directory:
            self.try_delete_dir(directory=self.directory)
        if exc_type is not None:
            raise exc_type(exc_val)

    def __repr__(self) -> str:
        return self.directory

    def __str__(self) -> str:
        return self.directory

    def close(self):
        self.__exit__(None, None, None)

    def close_timer(self, timer_seconds: int = 2):
        timer = Timer(timer_seconds, self.close)
        timer.start()


class TimerInline(object):
    def __init__(self, label: str = '', print_starting: bool = True, prettify_time: bool = True, round_to: int = 8):
        self.label = label
        self.print_starting = print_starting
        self.prettify_time = prettify_time
        self.round_to = round_to
        self.start_time = None
        self.end_time = None
        self._subtimes = {
            # Example
            # '<sub_time_label>': {
                # 'time': time.time(),
            # }
        }
        self._progress_bar_active = False

    def __enter__(self):
        if self.print_starting:
            prtcolor(text=f'Starting {self.label}...', color_code=35, prefix='[TIMER INLINE]', add_prefix=True)
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        elapsed = self.end - self.start
        if self._progress_bar_active:
            print()
        if self.prettify_time:
            text = f'{self.label} took {pretty_time(seconds=elapsed, round_to_small=self.round_to)} to complete'
        else:
            text = f'{self.label} took {elapsed:,.{self.round_to}f} seconds to complete'
        prtcolor(text=text, color_code=35, prefix='[TIMER INLINE]', add_prefix=True)
    
    def subtime(self, label: Optional[str] = None):
        if label is None:
            label = f'subtime'
        
        if label not in self._subtimes:
            self._subtimes[label] = {'time': self.start}
        
        now = time.time()
        pretty_sub_time = pretty_time(seconds=now - self._subtimes[label]['time'], round_to_small=self.round_to)
        prtcolor(
            text=f'Subtime "{label}" took {pretty_sub_time} to complete', 
            color_code=35, 
            prefix='[TIMER INLINE]', 
            add_prefix=True,
            add_newline=False
        )
        self._subtimes[label]['time'] = now
    
    def progress_bar(self, start_iter: int, at_iter: int, total_length: int):
        """
        This will print a progress bar to the console, it's meant to be used within a loop, and will keep
        a progress bar in the terminal that updates whenever the method is called. Beware of printing other
        info within the loop because it will cause the progress bar to move to the next line.
        Example:
            with TimerInline('Show Progress') as ti:
                for i in range(1000):
                    ti.progress_bar(start_iter=0, at_iter=i, total_length=1000)
                    time.sleep(0.01)
            
                    [OUTPUT] -> At 500 / 1,000 - [********************************.................................] (93.772 iters/sec)
        """
        self._progress_bar_active = True
        len_progress_bar = 60
        iters_completed = at_iter - start_iter
        pct_complete = int(((iters_completed / total_length) * len_progress_bar))
        pct_incomplete = len_progress_bar - pct_complete
        elapsed_time = time.time() - self.start
        if elapsed_time == 0:
            elapsed_time = 0.000001
        text = f"\rAt {at_iter:,.0f}/{total_length:,.0f} - {'[' + ('*' * pct_complete) + ('.' * pct_incomplete) + ']'}"
        text += '('
        if iters_completed > 0:            
            iters_per_sec = iters_completed / elapsed_time
            remaining_seconds = (total_length - iters_completed) / iters_per_sec
            iters_per_sec = f'{iters_per_sec:.3f} it/s'
            text += f'{iters_per_sec} - {pretty_time(seconds=remaining_seconds, round_to_small=1)} remain - '
        text += f'{pretty_time(seconds=elapsed_time, round_to_small=1)} elapsed)'
        text += '                '
        sys.stdout.write(text)
        sys.stdout.flush()