from .base import *
from ..datatypes import *



class Log(Base):
    # Metadata
    __tablename__ = 'log'

    # Database Fields ################################################################################################################################
    # Info Data #############################################################################################
    log_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        comment='The log UUID.'
    )

    # Notification Data #####################################################################################
    date_created: Mapped[datetime] = mapped_column(
        FlexibleDatetime,
        nullable=False,
        default=datetime.now,
        comment='The datetime of the log was created.'
    )
    log_type: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=False,
        comment=f'The type of the log either: (info, warning, error).'
    )
    message: Mapped[Optional[str]] = mapped_column(
        String(512),
        comment='The message if info or warning.'
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        comment='The user_id of the affected user.'
    )
    func: Mapped[Optional[str]] = mapped_column(
        String(128),
        comment='The function name that the error log came from.'
    )
    exc_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        comment='The name of the Python exception.'
    )
    exc_args: Mapped[Optional[str]] = mapped_column(
        String(512),
        comment='The args of the Python exception.'
    )
    traceback: Mapped[Optional[str]] = mapped_column(
        String(4098),
        comment='The full traceback string of the exception.'
    )


    # Logging Attributes #############################################################################################################################
    ERROR_HTML_EMAIL = """
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
            {error_html_block}
        </body>
    </html>
    """

    ERROR_HTML_BLOCK = """
    <p>
        <strong>A Server Side exception has been raised from the MBGTools Application.</strong>
    </p>
    <p>
        <strong>Details:</strong><br><br>
        <table>
            <tbody>
                <tr>
                    <td style="width: 250px">Time of Error</td>
                    <td>{time_of_error}</td>
                </tr>
                <tr>
                    <td>App Function</td>
                    <td>{func}</td>
                </tr> 
                <tr>
                    <td>User Email</td>
                    <td>{user_email}</td>
                </tr>     
                <tr>
                    <td>Exception</td>
                    <td>{exc_name}</td>
                </tr>   
                <tr>
                    <td>Exception Args</td>
                    <td>{exc_args}</td>
                </tr>               
            </tbody>                
        </table>
    </p>
    <p>
        <strong>Message:</strong><br><br>
        {message}
    <p>
    <strong>Traceback:</strong><br><br>
    <code>{traceback}</code>
    </p>
    """

    # Constructor ####################################################################################################################################
    def __init__(
            self,
            log_type: Literal['info', 'warning', 'error'],
            message: Optional[str] = None,
            user_id: Optional[str] = None,
            func: Optional[str] = None,
            exc_name: Optional[str] = None,
            exc_args: Optional[str] = None,
            traceback: Optional[str] = None
    ):
        self.log_type = log_type
        self.message = message
        self.func = func
        self.user_id = user_id
        self.exc_name = exc_name
        self.exc_args = exc_args
        self.traceback = traceback
        super().__init__()

    @classmethod
    def _log(
            cls,
            log_type: Literal['info', 'warning', 'error'],
            message: Optional[str] = None,
            user_id: Optional[str] = None,
            func: Optional[str] = None,
            exc_name: Optional[str] = None,
            exc_args: Optional[str] = None,
            traceback: Optional[str] = None,
            send_email_on_error: bool = False,
            user_email: Optional[str] = None,
    ):
        if send_email_on_error and log_type in ['error', 'warning'] and not isna(user_email):
            send_email(
                to=[EnvReader.get('MAIL_STEWARD'), user_email],
                subject='Trader App Error',
                message=cls.ERROR_HTML_EMAIL.format(
                    error_html_block=cls.ERROR_HTML_BLOCK.format(
                        time_of_error=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        func=func,
                        user_email=user_email,
                        exc_name=exc_name,
                        exc_args=exc_args,
                        message=message,
                        traceback=traceback
                    )
                ),
                message_contains_html=True
            ) 

        return cls(
            log_type=log_type,
            message=message,
            user_id=user_id,
            func=func,
            exc_name=exc_name,
            exc_args=exc_args,
            traceback=traceback
        )  

    @classmethod
    def info(
        cls,
        message: Optional[str] = None,
        user_id: Optional[str] = None,
        func: Optional[str] = None,
        send_email_on_error: bool = False,
        user_email: Optional[str] = None,
    ) -> 'Log':
        log = cls._log(
            log_type='info', 
            message=message,
            user_id=user_id,
            func=func,
            send_email_on_error=send_email_on_error,
            user_email=user_email
        )
        prtcolor(text=f'[{cls.__class__.__name__} INFO]: {message}', color_code=35)
        return log

    @classmethod
    def warn(        
        cls,
        message: Optional[str] = None,
        user_id: Optional[str] = None,
        func: Optional[str] = None,
        send_email_on_error: bool = False,
        user_email: Optional[str] = None,
    ) -> 'Log':
        log = cls._log(
            log_type='warning', 
            message=message,
            user_id=user_id,
            func=func,
            send_email_on_error=send_email_on_error,
            user_email=user_email
        )
        prtcolor(text=f'[{cls.__class__.__name__} WARNING]: {message}', color_code=35)
        return log

    @classmethod
    def error(
            cls,
            exception: Exception,
            traceback: str,
            message: Optional[str] = None,
            user_id: Optional[str] = None,
            func: Optional[str] = None,
            send_email_on_error: bool = False,
            user_email: Optional[str] = None,
    ) -> 'Log':
        exc_name = exception.__class__.__name__
        exc_args = str(exception.args)
        # Print the error to the console before the traceback gets updated
        prtcolor(text=f'[{cls.__class__.__name__} ERROR]: {exc_name} {exc_args}\nTraceback:\n{traceback}', color_code=35)
        traceback = traceback.replace('\n', '<br>')
        if len(traceback) > 4098:  # Limit the length due to database field size
            traceback = traceback[:2032] + '<br><br>... TRUNCATED ...<br><br>' + traceback[-2032:]
        log = cls._log(
            log_type='error',
            message=message,
            exc_name=exc_name,
            exc_args=exc_args,
            traceback=traceback,
            user_id=user_id,
            func=func,
            send_email_on_error=send_email_on_error,
            user_email=user_email
        )
        return log