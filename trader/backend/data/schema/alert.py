
from .base import *
from ..datatypes import *



class Alert(Base):
    # Metadata
    __tablename__ = 'alert'

    # Database Fields ################################################################################################################################
    # Info Data #############################################################################################
    alert_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        comment='The alert UUID.'
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey('user.user_id', ondelete='CASCADE', onupdate='CASCADE'),
        comment='The user the alert belongs to'
    )

    # Notification Data #####################################################################################
    date_sent: Mapped[datetime] = mapped_column(
        FlexibleDatetime,
        nullable=False,
        default=datetime.now,
        comment='The datetime the notification was sent'
    )
    subject: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment='The alert subject.'
    )
    body: Mapped[str] = mapped_column(
        String(4096),
        nullable=False,
        comment='The alert body.'
    )

    # Relationships #########################################################################################
    # Parent
    user: Mapped['User'] = relationship(back_populates='alerts', **Base.__child_relationship_kwargs__)

    # Constructor ####################################################################################################################################
    def __init__(
            self,
            user: User,
            subject: str,
            body: str
    ):
        self.user = user
        self.user_id = self.user.user_id
        self.subject = subject
        self.body = body
        super().__init__()