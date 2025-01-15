from .base import *
from ..datatypes import *



class Trade(Base):
    # Metadata
    __tablename__ = 'trade'

    trade_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        comment='The trade UUID.'
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey('user.user_id', ondelete='CASCADE', onupdate='CASCADE'),
        nullable=False,
        comment='The user the trade belongs to'
    )
    ticker_id: Mapped[str] = mapped_column(
        ForeignKey('ticker.ticker_id', ondelete='CASCADE', onupdate='CASCADE'),
        nullable=False,
        comment='The ticker the trade was executed on.'
    )

    date_executed: Mapped[datetime] = mapped_column(
        FlexibleDatetime,
        nullable=False,
        default=datetime.now,
        comment='The datetime the trade was executed'
    )

    trade: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment='The type of trade (buy or sell).'
    )

    shares: Mapped[float] = mapped_column(
        FloatNumpy,
        nullable=False,
        comment='The number of shares traded.'
    )

    price: Mapped[float] = mapped_column(
        FloatNumpy,
        nullable=False,
        comment='The price of the trade.'
    )

    # Relationships #########################################################################################
    # Parent
    user: Mapped['User'] = relationship(back_populates='trades', **Base.__child_relationship_kwargs__)
    ticker: Mapped['Ticker'] = relationship(back_populates='trades', **Base.__child_relationship_kwargs__)


    # Constructor ####################################################################################################################################
    def __init__(
        self,
        user: 'User',
        ticker: 'Ticker',
        trade: Literal['buy', 'sell'],
        shares: float,
        price: float,
        date_executed: Optional[Union[datetime, date, str, pd.Timestamp]] = None
    ):
        self.user = user
        self.user_id = self.user.user_id
        self.ticker = ticker
        self.ticker_id = self.ticker.ticker_id
        self.trade = trade
        self.shares = shares
        self.price = price
        if not isna(date_executed):
            self.date_executed = date_executed
        super().__init__()