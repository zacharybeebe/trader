from abc import ABC
from ..imports import *


# Custom Type Decorators ############################################################################################################################
class BooleanNumpy(TypeDecorator):
    impl = Boolean
    cache_ok = True

    def process_bind_param(self, value: Optional[Union[int, float]], dialect) -> Optional[float]:
        return convert_numpy_value(value)

    def process_result_value(self, value: float, dialect) -> Optional[int]:
        return value


class IntegerNumpy(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value: Optional[Union[int, float]], dialect) -> Optional[float]:
        return convert_numpy_value(value)

    def process_result_value(self, value: float, dialect) -> Optional[int]:
        return value


class FloatNumpy(TypeDecorator):
    impl = Float
    cache_ok = True

    def process_bind_param(self, value: Optional[Union[int, float]], dialect) -> Optional[float]:
        return convert_numpy_value(value)

    def process_result_value(self, value: float, dialect) -> Optional[int]:
        return value
    
    
class FlexibleDatetime(TypeDecorator):
    """
    FlexibleDatetime is a custom SQLAlchemy datatype that is able to accept string, datetime, date, and pd.Timestamp objects.
    This will convert them into Python datetime objects before storing them in the database.
    """
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[Union[str, datetime, date, pd.Timestamp]], dialect) -> Optional[datetime]:
        return datetime_parse(dt=value, nan_to_now=False)

    def process_result_value(self, value: Optional[str], dialect) -> Optional[list]:
        if isna(value):
            return None
        else:
            return value
    
    
class UTCDatetime(TypeDecorator):
    """
    FlexibleDatetime is a custom SQLAlchemy datatype that is able to accept string, datetime, date, and pd.Timestamp objects.
    This will convert them into Python datetime objects before storing them in the database.
    """
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[Union[str, datetime, date, pd.Timestamp]], dialect) -> Optional[datetime]:
        parsed = datetime_parse(dt=value, nan_to_now=False)
        if parsed is not None:
            utc_dt = datetime(
                year=parsed.year, 
                month=parsed.month, 
                day=parsed.day, 
                hour=parsed.hour, 
                minute=parsed.minute, 
                second=parsed.second, 
                tzinfo=timezone.utc
            )
            return utc_dt
        else:
            return None

    def process_result_value(self, value: Optional[str], dialect) -> Optional[list]:
        if isna(value):
            return None
        else:
            return datetime(
                year=value.year, 
                month=value.month, 
                day=value.day, 
                hour=value.hour, 
                minute=value.minute, 
                second=value.second, 
                tzinfo=timezone.utc
            )


class Password(TypeDecorator):
    impl = String
    cache_ok = True

    def process_bind_param(self, value: Optional[str], dialect) -> Optional[str]:
        # Passwords will be stored as sha256 hashes
        if isna(value):
            return None
        else:
            return sha256(value.encode('utf-8')).hexdigest()

    def process_result_value(self, value: str, dialect) -> str:
        return value