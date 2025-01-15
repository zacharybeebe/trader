import pandas as pd
import talib

from functools import wraps
from typing import Optional


class Candles(object):
    @classmethod
    def all_candle_names(cls):
        return [attr[4:] for attr in cls.__dict__ if attr.startswith('cdl_')]
    
    @classmethod
    def all_candle_func_names(cls):
        return [attr for attr in cls.__dict__ if attr.startswith('cdl_')]

    @staticmethod
    def _wrap_candle(func):
        @wraps(func)
        def wrapper(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
            candle_method = getattr(talib, func.__name__.replace('_', '').upper())
            series = candle_method(data['open'], data['high'], data['low'], data['close'])
            if not name_with_cdl_prefix:
                series.name = func.__name__[4:]
            elif custom_name:
                series.name = custom_name
            else:
                series.name = func.__name__
            return series
        return wrapper

    # Candle Methods will be wrapped by _wrap_candle, which will do the work of applying the candle to the DataFrame
    # The actual candle methods will just pass, because they will go directly to the _wrap_candle method

    # While there are some typically more Bullish or Bearish candles, TA-Lib doesn't differentiate in that way; instead it
    # will return 100 if there is a bullish trend, and -100 if there is a bearish trend, 0 is the default and means no trend

    @staticmethod
    @_wrap_candle
    def cdl_2_crows(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Two Crows"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_3_black_crows(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Three Black Crows"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_3_inside(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Three Inside Up/Down"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_3_linestrike(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Three Outside Up/Down"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_3_stars_in_south(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Three Stars In The South"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_3_white_soldiers(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Three Advancing White Soldiers"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_abandoned_baby(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Abandoned Baby"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_advance_block(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Advance Block"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_belt_hold(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Belt-hold"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_breakaway(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Breakaway"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_closing_marubozu(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Closing Marubozu"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_conceal_baby_swall(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Concealing Baby Swallow"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_counterattack(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Counterattack"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_dark_cloud_cover(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Dark Cloud Cover"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_doji(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Doji"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_doji_star(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Doji Star"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_dragonfly_doji(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Dragonfly Doji"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_engulfing(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Engulfing Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_evening_doji_star(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Evening Doji Star"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_evening_star(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Evening Star"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_gap_side_side_white(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Up/Down-gap side-by-side white lines"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_gravestone_doji(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Gravestone Doji"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_hammer(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Hammer"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_hanging_man(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Hanging Man"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_harami(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Harami Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_harami_cross(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Harami Cross Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_high_wave(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """High-Wave Candle"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_hikkake(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Hikkake Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_hikkake_mod(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Modified Hikkake Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_homing_pigeon(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Homing Pigeon"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_identical_3_crows(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Identical Three Crows"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_in_neck(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """In-Neck Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_inverted_hammer(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Inverted Hammer"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_kicking(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Kicking"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_kicking_by_length(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Kicking - bull/bear determined by the longer marubozu"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_ladder_bottom(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Ladder Bottom"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_long_legged_doji(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Long Legged Doji"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_long_line(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Long Line Candle"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_marubozu(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Marubozu"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_matching_low(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Matching Low"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_mat_hold(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Mat Hold"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_morning_doji_star(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Morning Doji Star"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_morning_star(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Morning Star"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_on_neck(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """On-Neck Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_piercing(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Piercing Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_rickshaw_man(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Rickshaw Man"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_rise_fall_3_methods(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Rising/Falling Three Methods"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_separating_lines(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Separating Lines"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_shooting_star(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Shooting Star"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_short_line(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Short Line Candle"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_spinning_top(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Spinning Top"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_stalled_pattern(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Stalled Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_stick_sandwich(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Stick Sandwich"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_takuri(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Takuri (Dragonfly Doji with very long lower shadow)"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_tasukigap(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Tasuki Gap"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_thrusting(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Thrusting Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_tristar(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Tristar Pattern"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_unique_3_river(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Unique 3 River"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_upside_gap_2_crows(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Upside Gap Two Crows"""
        pass

    @staticmethod
    @_wrap_candle
    def cdl_xside_gap_3_methods(data: pd.DataFrame, name_with_cdl_prefix: bool = False, custom_name: Optional[str] = None) -> pd.Series:
        """Upside/Downside Gap Three Methods"""
        pass