from .base import *
from ..datatypes import *



class Interval(Base):
    __tablename__ = 'interval'

    interval_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        comment='The interval UUID.'
    )
    ticker_id: Mapped[str] = mapped_column(
        ForeignKey('ticker.ticker_id', ondelete='CASCADE', onupdate='CASCADE'),
        nullable=False,
        comment='The ticker the interval belongs to.'
    )

    interval: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment='The interval string (1m, 5m, 1d, etc.).'
    )

    # Relationships #########################################################################################
    # Parent
    ticker: Mapped['Ticker'] = relationship(back_populates='intervals', **Base.__child_relationship_kwargs__)
    # Children
    candles: Mapped[List['Candle']] = relationship(back_populates='interval', **Base.__parent_relationship_kwargs__)

    def __init__(self, ticker: 'Ticker', interval: str):
        self.ticker = ticker
        self.ticker_id = self.ticker.ticker_id
        self.interval = interval
        super().__init__()