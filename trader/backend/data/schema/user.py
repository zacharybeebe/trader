from .base import *
from ..datatypes import *



class User(Base):
    __tablename__ = 'user'

    user_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        comment='The user UUID.'
    )

    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment='The user name.'
    )

    email: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment='The user email.'
    )

    password: Mapped[str] = mapped_column(
        Password,
        nullable=False,
        comment='The user password.'
    )

    rh_email: Mapped[bytes] = mapped_column(
        LargeBinary,
        nullable=False,
        comment='The user Robinhood email, encrypted.'
    )

    rh_password: Mapped[bytes] = mapped_column(
        LargeBinary,
        nullable=False,
        comment='The user Robinhood password, encrypted.'
    )

    # Relationships #########################################################################################
    # Children
    alerts: Mapped[List['Alert']] = relationship(back_populates='user', **Base.__parent_relationship_kwargs__)
    trades: Mapped[List['Trade']] = relationship(back_populates='user', **Base.__parent_relationship_kwargs__)


    def __init__(self, name: str, email: str, password: str, rh_email: bytes, rh_password: bytes):
        self.name = name
        self.email = email
        self.password = password
        self.rh_email = rh_email
        self.rh_password = rh_password
        super().__init__()