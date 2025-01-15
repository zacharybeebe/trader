from .base import *
from ..datatypes import *



class Ticker(Base):
    __tablename__ = 'ticker'

    ticker_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        comment='The ticker UUID.'
    )

    symbol: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment='The ticker symbol.'
    )

    # Relationships #########################################################################################
    # Children
    intervals: Mapped[List['Interval']] = relationship(back_populates='ticker', **Base.__parent_relationship_kwargs__)
    trades: Mapped[List['Trade']] = relationship(back_populates='ticker', **Base.__parent_relationship_kwargs__)

    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__()

