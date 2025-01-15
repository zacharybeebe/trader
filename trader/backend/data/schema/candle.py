from .base import *
from ..datatypes import *



class Candle(Base):
    __tablename__ = 'candle'

    candle_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        comment='The candle UUID.'
    )
    interval_id: Mapped[str] = mapped_column(
        ForeignKey('interval.interval_id', ondelete='CASCADE', onupdate='CASCADE'),
        nullable=False,
        comment='The interval the candle record belongs to.'
    )

    dt: Mapped[datetime] = mapped_column(
        UTCDatetime,
        nullable=False,
        comment='The datetime of the price info.'
    )

    open: Mapped[float] = mapped_column(
        FloatNumpy,
        nullable=False,
        comment='The open price.'
    )

    high: Mapped[float] = mapped_column(
        FloatNumpy,
        nullable=False,
        comment='The high price.'
    )

    low: Mapped[float] = mapped_column(
        FloatNumpy,
        nullable=False,
        comment='The low price.'
    )

    close: Mapped[float] = mapped_column(
        FloatNumpy,
        nullable=False,
        comment='The close price.'
    )

    volume: Mapped[float] = mapped_column(
        FloatNumpy,
        nullable=False,
        comment='The volume of the price.'
    )

    # Relationships #########################################################################################
    # Parent
    interval: Mapped['Interval'] = relationship(back_populates='candles', **Base.__child_relationship_kwargs__)


    def __init__(self, interval: 'Interval', dt: datetime, open: float, high: float, low: float, close: float, volume: float):
        self.interval = interval
        self.interval_id = self.interval.interval_id
        self.dt = dt
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        super().__init__()