from .schema.base import *


class Engine(object):
    __BASE__ = Base

    class SchemaClassHolder(object):
        def __init__(self, base: Optional[DeclarativeBase] = None):
            self.__base__ = base

            # Schema Table Classes
            self.log = Log
            self.user = User
            self.alert = Alert
            self.ticker = Ticker
            self.trade = Trade
            self.interval = Interval
            self.candle = Candle

        def __getitem__(self, item):
            return getattr(self, item)

        def __iter__(self):
            if self.__base__ is None:
                return iter([])
            else:
                return iter(self.__dict__.values())

        def hierarchical_table_names(self, association_tables_last: bool = True) -> list:
            """
            The __base__ object will have an attached metadata object that will contain the tables in their hierarchical order.
            If "association_tables_last" is True, then the association tables will be at the end of the list, otherwise they will be
            at the beginning, per the SQLAlchemy convention.
            """
            if self.__base__ is None:
                return []

            hierarchical_tables = []
            association_tables = []
            for table_name in self.__base__.metadata.tables.keys():
                if table_name.startswith('table_') and association_tables_last:
                    association_tables.append(table_name)
                else:
                    hierarchical_tables.append(table_name)
            return hierarchical_tables + association_tables

    # t attribute is the shorthand for accessing the mbg classes (RMUSchema) -> self.t.rmu = RMUSchema
    t = SchemaClassHolder(base=__BASE__)

    class Transaction:
        def __init__(
                self,
                engine: 'Engine',
                hold_commit_until_close: bool = True,
                rollback_on_error: bool = True,

                # Session specific kwargs
                expire_on_commit: bool = False,
                autobegin: bool = True,
                twophase: bool = False,
                enable_baked_queries: bool = True,
                info: Optional[dict] = None,
                user_id: Optional[str] = None
        ):
            """
            Transactions act as a holder for a SQLAlchemy session, but are meant to persist through multiple method or function calls.
            Args:
                engine:
                hold_commit_until_close:
                rollback_on_error:
                expire_on_commit:
                autobegin:
                twophase:
                enable_baked_queries:
                info:
            """
            self._id = generate_uuid()
            self._engine = engine
            self.session: Session = self._engine.make_session(
                expire_on_commit=expire_on_commit,
                autobegin=autobegin,
                twophase=twophase,
                enable_baked_queries=enable_baked_queries,
                info=info
            )
            self._hold_commit_until_close = hold_commit_until_close
            if self._hold_commit_until_close:
                self.auto_commit = False
            else:
                self.auto_commit = True
            self.rollback_on_error = rollback_on_error
            self._engine.active_transactions[self._id] = self
            self.user_id = user_id

        def __repr__(self):
            repr = f'<{self.__class__.__name__}\n\t'
            joins = []
            for k, v in self.__dict__.items():
                if k == '_id' or not k.startswith('_'):
                    joins.append(f'{k:20}{v}')
            repr += '\n\t'.join(joins)
            repr += '\n>'
            return repr

        def close(self, expunge_all: bool = False):
            if self._hold_commit_until_close:
                self.session.commit()
            if expunge_all:
                self.session.expunge_all()
            self.session.close()
            self._engine.active_transactions.pop(self._id, None)

        def close_all(self, expunge_all: bool = False):
            all_transactions = list(self._engine.active_transactions.values())
            for trx in all_transactions:
                trx.close(expunge_all=expunge_all)
            self.session.close_all()

        def commit(self):
            self.session.commit()

        def expunge(self, instance):
            self.session.expunge(instance)

        def expunge_all(self):
            self.session.expunge_all()

        def expunge_many(self, instances):
            for instance in instances:
                self.session.expunge(instance)

    _IGNORE_DEFAULT_TABLES = ['spatial_ref_sys']

    _SQL_QUOTE_PATTERN = re.compile(r"""[\s]{1}[\"\']{1}(.*?)[\"\'][\s]{1}""", flags=re.MULTILINE)
    _CONSTRAINT_DEPENDENT_PATTERN = re.compile(r"""[\(]?([\w]+[\ ]*=[\ ]*[\'\"]{1}[\w\ ]+[\'\"]{1})[\ ]+AND[\ ]+([\w]+[\ ]+IN[\ ]+[\(]{1}[\w\"\'\,\.\ ]+[\)]{1})[\)]?[\ ]*[OR[\ ]*]?""", flags=re.MULTILINE)
    _CONSTRAINT_VALUES_PATTERN = re.compile(r"""([\w]+[\ ]+IN[\ ]+[\(]{1}[\w\"\'\,\.\ ]+[\)]{1})""", flags=re.MULTILINE)

    _TICKER_PAIRS = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'LTC': 'LTC-USD',
        'DOGE': 'DOGE-USD',
    }

    # Exceptions #####################################################################################################################################    
    class CannotCopyDatabaseError(Exception):
        def __init__(self, current_engine_name: str, copy_to_engine_name: str):
            super().__init__(f'The Engine to copy to must be an instance of the same DatabaseEngine class or child class, <{current_engine_name}> != <{copy_to_engine_name}>')

    class DeclarativeBaseError(ValueError):
        def __init__(self, method_name: str):
            msg = f'This instance does not have "__BASE__" class-attribute set as a DeclarativeBase (or subclass of DeclarativeBase), cannot perform requested action of "{method_name}".'
            super().__init__(msg)

    class InvalidDatabaseType(Exception):
        def __init__(self, database_type: str):
            super().__init__(f"Invalid database_type argument: '{database_type}'. Please choose from {sqlize_type_literal(Typing.Shared.DATABASE_TYPES)}")

    class TableNameNotFound(Exception):
        def __init__(self, table_name: str, current_table_names: list):
            super().__init__(f'Could not locate "table_name": "{table_name}". Current Tables in Database are: {current_table_names}')

    # Constructor ####################################################################################################################################
    def __init__(
            self,
            database_type: Literal['sqlite', 'postgres'],
            engine_echo: bool = True,
            show_description: bool = True,
            drop_all_tables_on_init: bool = False,
            create_all_tables_on_init: bool = False,
            disable_dynamic_table_updater: bool = False,
            **connection_kwargs,
    ):
        self.database_type = database_type
        self.fernet_encryption_key = EnvReader.get('ENCRYPTION_KEY')
        self.active_transactions = {}

        if self.database_type == 'sqlite':
            self.description = 'DEVL SQLite Database'
            self.filepath = os.path.join(os.path.dirname(__file__), 'database.db')
            self.filepath = self.filepath.replace('\\', '/')
            self.file_ext = self.filepath.split('.')[-1]

            self.username = None
            self.password = None
            self.host = None
            self.port = None
            self.database_name = None
            self.schema = None

        else:
            self.description = 'PROD Postgres Database'
            self.filepath = None
            self.file_ext = None

            self.username = EnvReader.get('DATABASE_USERNAME')
            self.password = EnvReader.get('DATABASE_PASSWORD')
            self.host = EnvReader.get('DATABASE_HOST')
            self.port = EnvReader.get('DATABASE_PORT')
            self.database_name = EnvReader.get('DATABASE_NAME')
            self.schema = 'public'

        self._do_drop_all_tables = drop_all_tables_on_init
        self._do_create_all_tables = create_all_tables_on_init

        if self.description is not None and show_description:
            self.prtinfo(text=self.description)

        self.engine: Optional[Engine] = None

        if self.database_type == 'sqlite':
            if not os.path.isfile(self.filepath):
                if drop_all_tables_on_init:
                    try_remove_file(self.filepath)
                self.prtinfo(text=f'Creating new SQLite database at "{self.filepath}"')
                self._do_create_all_tables = True
            self.connection_credentials = f"sqlite:///{os.path.abspath(self.filepath)}"

        elif self.database_type == 'postgres':
            if self.port is None:
                self.connection_credentials = f'postgresql://{self.username}:{self.password}@{self.host}/{self.database_name}'
            else:
                self.connection_credentials = f'postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}'

        else:
            raise self.InvalidDatabaseType(database_type=self.database_type)

        engine_kwargs = filter_callable_kwargs(
            func=create_engine,
            passed_kwargs=connection_kwargs,
            remove_predefined_kwargs=['url', 'echo']
        )
        if self.schema is not None:
            # Add mbg to engine
            if 'execution_options' in engine_kwargs:
                engine_kwargs['execution_options'].update({'schema_translate_map': {None: self.schema}})
            else:
                engine_kwargs['execution_options'] = {'schema_translate_map': {None: self.schema}}
            self.engine = create_engine(
                url=self.connection_credentials,
                echo=engine_echo,
                **engine_kwargs
            )
        else:
            self.engine = create_engine(url=self.connection_credentials, echo=engine_echo, **engine_kwargs)

        # Attach events now that the engine is created
        self.is_closed: bool = False
        atexit.register(self._at_exit_cleanup)

        # Drop and/or create all tables, if desired from the init args
        if self.__BASE__ is not None:
            if self._do_drop_all_tables:
                self.drop_all_tables_from_base()

            if self._do_create_all_tables:
                self.create_all_tables_from_base()

        # Get table names by inspecting self.engine - pass schema if it is not None
        inspector = sa_inspect(self.engine)
        if self.schema is not None:
            self.table_names = [i for i in inspector.get_table_names(schema=self.schema) if i not in self._IGNORE_DEFAULT_TABLES]
        else:
            self.table_names = [i for i in inspector.get_table_names() if i not in self._IGNORE_DEFAULT_TABLES]

        # This method will look for new tables or columns defined from the DeclarativeBase schema classes
        # and either adds the new tables or adds new columns to an existing table
        if not disable_dynamic_table_updater:
            self._dynamic_table_updater()

    # Session Specific Methods that will be wrapped ##################################################################################################
    @staticmethod
    def wrap_session(
            func: Optional[Callable] = None,
            session: Optional[Session] = None,
            auto_commit: bool = True,
            rollback_on_error: bool = True,
            close_on_exit: bool = True,
            log_or_raise_on_error: Literal['log', 'raise'] = 'log',
            transaction: Optional['Engine.Transaction'] = None,
            timer: bool = False
    ):
        """
        This method is a decorator that will wrap a function with a session. It can be used with arguments or without:
            Example with args
                @wrap_session(session=session, auto_commit=True, rollback_on_error=True, close_on_exit=True, exception_handler=None, timer=False)
                def my_function(self, *args, **kwargs):
                    session=kwargs['session']
                    ...

            Example without args (in which the default args will be used):
                @wrap_session
                def my_function(self, *args, **kwargs):
                    session = kwargs['session']
                    ...

        A session may be passed into the decorator args, or in the function keyword arguments, if no session is passed in, a new session
        will be created by the instance of the DatabaseEngine. Either way a decorated function will always have an active SQLAlchemy Session
        object passed to it as the "session" keyword argument, along with the other default keyword arguments. Hence, it is REQUIRED that
        the decorated function has the **kwargs convention, otherwise Python will raise an exception.

        The decorator will handle committing or rolling back transactions as well as closing the session dependent on the given keyword arguments.
        If you would like to run multiple queries with the same session, you can start a transaction (DatabaseEngine.start_transaction(...))
        and pass it to the function, this transaction will contain the same keyword args  as the decorator (other than func, close_on_exit, and timer)
        and will supercede both the decorator and function args, if any are passed. This transaction is essentially a holder for different sessions
        in case asynchronous requests are coming in. start_transaction() will make a new session and persist it until Transaction.close() is called.
        Example:
            db = DatabaseEngine(...)
            transaction = db.start_transaction(auto_commit=True, rollback_on_error=True, logger=LoggingManager(...))
            # The session and keywords are passed by way of the transaction and wrap_session will use this transaction in the decorator
            x = db.query(..., transaction=transaction).first()
            y = db.query(..., transaction=transaction).first()
            z = db.query(..., transaction=transaction).first()
            transaction.close()

            # If you need to persist an ORM object after the session is closed, set the "expunge_all" argument to True within
            # the transaction.close() method - transaction.close(expunge_all=True). But be aware that the object will be somewhat detached from
            # the ORM so make sure load any attached ORM objects (parent, children, etc.) before closing the transaction.

        The function itself can also have the same keyword arguments as the decorator and the values of those function-arguments will
        take precedence over the decorator-arguments, example:
            @wrap_session
            def my_function(self, auto_commit=False, rollback_on_error=False, close_on_exit=False, **kwargs):
                session = kwargs['session']
                ...

        ** Note that this decorator can only decorate instance-methods of DatabaseEngine or its subclasses, as the self parameter is required **

        :param func:                The function to be wrapped, this should NOT be passed as an argument when using the decorator,
                                    the decorator is set up to find the function automatically.
        :param session:             And existing session object to be used, if None, a new session will be created
        :param auto_commit:         Boolean indicating if the session should be committed after the function completes
        :param rollback_on_error:   Boolean indicating if the session should be rolled back if an exception occurs
        :param close_on_exit:       Boolean indicating if the session should be closed after the function completes
        :param logger:              An optional instance of the BKDatabaseSession.LoggingManager class to be used for logging exceptions
        :param transaction:         An optional instance of the DatabaseEngine.Transaction class to be used for persisted sessions
        :param timer:               Boolean indicating if the time to complete the function should be printed to the console
        :return:
        """
        # Helper Functions #######################################################################################
        def update_session_kwargs(self, **kw):
            trx = kw.get('transaction', transaction)
            if trx is not None:
                kw['session'] = trx.session
                kw['auto_commit'] = trx.auto_commit
                kw['rollback_on_error'] = trx.rollback_on_error
                kw['user_id'] = trx.user_id
            else:
                kw['session'] = kw.get('session', session)
                if kw['session'] is None:
                    kw['session'] = self.make_session()
                kw['auto_commit'] = kw.get('auto_commit', auto_commit)
                kw['rollback_on_error'] = kw.get('rollback_on_error', rollback_on_error)
                kw['close_on_exit'] = kw.get('close_on_exit', close_on_exit)
                kw['user_id'] = None
            return kw

        def run_function(ff, self, *a, **k):
            new_kwargs = update_session_kwargs(self, **k)
            try:
                if timer:
                    now = perf_counter()
                results = ff(self, *a, **new_kwargs)
                if new_kwargs['auto_commit']:
                    new_kwargs['session'].commit()
                if timer:
                    prtinfo(text=f'Function: {ff.__name__} took {perf_counter() - now:,.6f} seconds to complete')
                return results
            except Exception as e:
                if new_kwargs['rollback_on_error']:
                    new_kwargs['session'].rollback()

                # if log_or_raise_on_error == 'log':
                #     tb = traceback.format_exc()
                #     user_id = new_kwargs.get('user_id', None)
                #     user_email = None
                #     if user_id is not None:
                #         user_email = self.query(self.t.user).filter_by(user_id=user_id).first()
                #         if user_email is not None:
                #             user_email = user_email.email
                #     log = self.t.log.error(
                #         exception=e,
                #         traceback=tb,
                #         message=f'Error in function: {ff.__name__}',
                #         user_id=user_id,
                #         func=ff.__name__,
                #         send_email_on_error=False,  # Change to True if you want to send an email ###################################################
                #         user_email=user_email
                #     )
                #     self.add(log)
                # else:
                raise e
            finally:
                if new_kwargs.get('transaction', None) is None and new_kwargs['close_on_exit']:
                    new_kwargs['session'].close()

        # The decorator execution ################################################################################
        if callable(func):
            # The decorator has been assigned without keyword args: @wrap_session
            def wrapper(self, *args, **kwargs):
                # print(f'1: {func.__name__=}')
                # print(f'1: {self=}')
                # print(f'1: {args=}')
                # print(f'1: {kwargs=}\n')
                return run_function(func, self, *args, **kwargs)
            return wrapper
        else:
            # The decorator has been assigned with keyword args: @wrap_session(...)
            def decorator(f):
                def wrapper(self, *args, **kwargs):
                    # print(f'2: {f.__name__=}')
                    # print(f'2: {self=}')
                    # print(f'2: {args=}')
                    # print(f'2: {kwargs=}\n')
                    return run_function(f, self, *args, **kwargs)
                return wrapper
            return decorator

    @wrap_session
    def add(self, instance: object, _warn: bool = True, **session_kwargs) -> None:
        session = session_kwargs['session']
        session.add(instance, _warn)

    @wrap_session
    def add_all(self, instances: Iterable[object], **session_kwargs) -> None:
        session = session_kwargs['session']
        for instance in instances:
            session.add(instance)
    
    @wrap_session    
    def check_unique(self, table_name: str, field_name: str, field_value: Any, **session_kwargs) -> bool:
        """
        Checks if a field_value is unique in a given table. This method will return True if the field_value is unique, otherwise False.
        :param table_name:      The table_name in which to check for the unique field_value
        :param field_name:      The field_name for which to check the field_value
        :param field_value:     The field_value to check for uniqueness
        :return:    bool - True if the field_value is unique, otherwise False
        """
        is_unique = True

        df = self.qdf(
            sql=f"""
            SELECT
                {field_name}
            FROM
                $.{table_name}
            """,
            show_head=False,
            **session_kwargs
        )
        if df is not None:
            if field_value in df[field_name].values:
                is_unique = False
        return is_unique

    @wrap_session
    def delete(self, instance: object, **session_kwargs) -> None:
        session = session_kwargs['session']
        session.delete(instance)

    @wrap_session
    def delete_fast(self, table: DeclarativeBase, ids: list, **session_kwargs) -> None:
        """
        Sometimes when a schema class has a large depth of children or if there are many records to be deleted
        the orm delete method can be very slow. This method will uses raw SQL to do the deletes which is
        much faster. However because it is raw SQL, we have to also manually delete the records from association tables if there are any.
        """
        primary_key_field = self.primary_key_field(table=table)
        where_in_clause = sql_in_statement_check_length(iterable=ids, variable_name=primary_key_field)
        # Now delete the records from the main table
        self.qdf(
            sql=f"DELETE FROM $.{table.__table__.name} WHERE {where_in_clause}",
            warn_is_none=False,
            show_head=False,
            **session_kwargs
        )

    @wrap_session
    def qdf(
            self,
            sql: str,
            parameters: Optional[dict] = None,
            show_head: bool = True,
            null_zeros_for_columns: Optional[list] = None,
            index: Optional[Union[str, list]] = None,
            warn_is_none: bool = True,
            return_empty: bool = False,
            print_sql: bool = False,
            **session_kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Executes any user defined sql statement and if this sql statement returns data such as from a SELECT statement,
        the resulting data will be returned as a pandas DataFrame. Parameterized sql statements are accepted as well.

        For parameterized queries, use the syntax most associated with Postrges, MySQL, etc: ":myvariabe", NOT the SQLite syntax: "?" and
        pass in a dictionary of parameters with the keys being the placeholder variable names (without the colon) and their Python values.
        SQLAlchemy will handle the parameterization of the query the same for all database types.
            Examples:
                GOOD:  "SELECT * FROM table WHERE column = :myvariable"
                        parameters = {'myvariable': 'myvalue'}

                BAD:   "SELECT * FROM table WHERE column = ?"
                        parameters = ['myvalue']

        Note that if a query does not return data (as in the case of a modification statement) OR the query returns zero
        results the return value will be None, unless the "return_empty" argument is set to True, in which an empty DataFrame
        will be returned.

        Some columns may be saved in JSON format using the NestedMutableJson datatype from the sqlalchemy-json library. 
        Using the ORM, these attributes will automatically be converted into their primitave Python types when accessed, 
        HOWEVER because qdf() uses raw SQL, it is not aware of the different column types, so they need to be "manually" converted. To do this, 
        use the "json_columns_and_types" argument to pass in a dictionary with the column names as keys and the Python types as values.
        For example:
            db.qdf(
                ...,
                json_columns_and_types={'my_json_column': dict}
            )

        For predefined or constant-like queries or for queries that come from the app that can be for any database type, the table name
        within the query should be prepended with the special syntax '$.' (dollar sign dot), for example '$.strata'. This syntax will be replaced
        with the schema name if it is not None, otherwise it will be replaced with the empty string, leaving just the table name.
        This is important for Postges databases where schemas are a necessary fixture.

        :param sql:                     The sql statement to be executed
        :param parameters:              The parameters associated with a parameterized query
        :param json_columns_and_types:  A dictionary of column names and their associated Python types for columns that contain JSON data
        :param show_head:               Boolean indicating if the head of the resulting DataFrame should be printed to the console
        :param null_zeros_for_columns:  A list of column names for which zeros should be replaced with None
        :param index:                   Can be used to set the index of the resulting DataFrame
        :param warn_is_none:            Boolean indicating if a warning should be printed to the console when the query returns zero results
        :param return_empty:            Boolean indicating if an empty DataFrame should be returned when the query returns zero results, othwerwise None will be returned
        :param print_sql:               Boolean indicating if the final sql statement should be printed to the console
        :return: DataFrame or None
        """
        ##################################################################
        def null_zero(value: Any):
            if value == 0:
                return None
            else:
                return value
        ##################################################################
        session = session_kwargs['session']
        df = None
        # Check for the special syntax '$.' (dollar sign dot) and replace with the schema name
        # if it is not None, otherwise replace with empty string
        if '$.' in sql:
            if self.schema is not None:
                sql = sql.replace('$.', f'{self.schema}.')
            else:
                sql = sql.replace('$.', '')        

        # Check for single or double qoutes within the SQL statement and replace them with doubled quotes
        #   For example if a person has a name like 'O'Conner" the SQL statement will need to be 'O''Conner'
        quote_matches = re.finditer(self._SQL_QUOTE_PATTERN, sql)
        if quote_matches:
            for match in quote_matches:
                found = match.group(1)
                if ("'" in found):
                    new_found = found.replace("'", "''")
                    sql = sql.replace(found, new_found)
                
                elif ('"' in found):
                    new_found = found.replace('"', '""')
                    sql = sql.replace(found, new_found)

        sql = text(sql)
        if print_sql:
            print(f'{sql}\n')
        if parameters is not None:
            executed = session.execute(sql, parameters)
        else:
            executed = session.execute(sql)

        if executed.returns_rows:
            try:
                df = pd.DataFrame(data=executed.fetchall(), columns=list(executed.keys()))
            except ValueError:
                pass  # df defaults to None
            finally:
                executed.close()

        if df is None or len(df) == 0:
            if return_empty:
                return df
            else:
                if warn_is_none:
                    self.prtwarn(f'Query returned zero results, return object will be None')
                return None
        else:
            if index is not None:
                df.set_index(index, inplace=True)
            if show_head:
                print(f'{df.head()}\n')

            if null_zeros_for_columns is not None:
                for column in null_zeros_for_columns:
                    if column in df.columns:
                        df[column] = df[column].apply(func=null_zero)
            return df

    @wrap_session
    def query(self, entity: Type[DeclarativeBase], **session_kwargs):
        session = session_kwargs['session']
        return session.query(entity)

    @wrap_session
    def query_by_id(self, entity: Type[DeclarativeBase], orm_id: str, **session_kwargs) -> Optional[DeclarativeBase]:
        session = session_kwargs['session']
        return session.query(entity).filter_by(**{f'{entity.__table__.name}_id': orm_id}).first()
    
    @wrap_session
    def query_joined(self, entity: Type[DeclarativeBase], default_load_for: Optional[list] = None, **session_kwargs):
        return self._base_query_load_method(entity=entity, load_method=joinedload, default_load_for=default_load_for, **session_kwargs)
    
    @wrap_session
    def query_lazy(self, entity: Type[DeclarativeBase], default_load_for: Optional[list] = None, **session_kwargs):
        return self._base_query_load_method(entity=entity, load_method=lazyload, default_load_for=default_load_for, **session_kwargs)
    
    @wrap_session
    def query_noload(self, entity: Type[DeclarativeBase], default_load_for: Optional[list] = None, **session_kwargs):
        return self._base_query_load_method(entity=entity, load_method=noload, default_load_for=default_load_for, **session_kwargs)
    
    @wrap_session
    def query_raiseload(self, entity: Type[DeclarativeBase], default_load_for: Optional[list] = None, **session_kwargs):
        return self._base_query_load_method(entity=entity, load_method=raiseload, default_load_for=default_load_for, **session_kwargs)

    @wrap_session
    def query_selectin(self, entity: Type[DeclarativeBase], default_load_for: Optional[list] = None, **session_kwargs):
        return self._base_query_load_method(entity=entity, load_method=selectinload, default_load_for=default_load_for, **session_kwargs)
    
    @wrap_session
    def query_subquery(self, entity: Type[DeclarativeBase], default_load_for: Optional[list] = None, **session_kwargs):
        return self._base_query_load_method(entity=entity, load_method=subqueryload, default_load_for=default_load_for, **session_kwargs)

    @wrap_session
    def update(self, instance: object, field_values: dict, exclude_fields: list = [], **session_kwargs) -> None:
        for field, value in field_values.items():
            if hasattr(instance, field) and field not in exclude_fields:
                try:
                    setattr(instance, field, value)
                except (AttributeError, TypeError):
                    # Some attributes may be unsettable properties, so we skip them here
                    pass
    
    @wrap_session
    def get_ticker_interval_id(
        self,
        ticker: str,
        interval: T.Trade.INTERVAL,
        **session_kwargs
    ) -> Optional[tuple[str, str]]:
        
        ticker_interval = self.qdf(
            sql=f"""
            SELECT
                TCK.ticker_id,
                ITV.interval_id
            FROM
                $.interval ITV
            LEFT JOIN (
                SELECT
                    ticker_id,
                    symbol
                FROM
                    $.ticker
                WHERE
                    symbol = '{ticker.upper()}'
            ) TCK ON ITV.ticker_id = TCK.ticker_id
            WHERE
                TCK.ticker_id IS NOT NULL
                AND ITV.interval = '{interval.lower()}'
            """,
            show_head=False,
            **session_kwargs
        )
        if ticker_interval is None:
            return None
        else:
            return ticker_interval['ticker_id'].values[0], ticker_interval['interval_id'].values[0]
    
    @wrap_session
    def data(
        self, 
        ticker: str, 
        interval: T.Trade.INTERVAL, 
        start: Optional[datetime] = None, 
        end: Optional[datetime] = None, 
        **session_kwargs
    ) -> Optional[pd.DataFrame]:
        """
        This method will return the data for a given ticker and interval between the start and end dates. The data will be returned as a pandas DataFrame.
        """
        self.update_data(ticker=ticker, interval=interval, **session_kwargs)
        ticker_interval = self.get_ticker_interval_id(ticker=ticker, interval=interval, **session_kwargs)
        if ticker_interval is None:
            return None
                
        interval_id = ticker_interval[-1]
        candle_sql = f"""
        SELECT
            *
        FROM
            $.candle
        WHERE
            interval_id = '{interval_id}'
        """
        params = {}
        if not isna(start):
            candle_sql += f" AND dt >= :start"
            params['start'] = start
        if not isna(end):
            candle_sql += f" AND dt <= :end"
            params['end'] = end
        candle_sql += " ORDER BY dt ASC"

        df = self.qdf(
            sql=candle_sql,
            parameters=params if len(params) > 0 else None,
            show_head=False,
            **session_kwargs
        )
        # Convert the dt column to a datetime object
        if df is not None:
            df['dt'] = pd.to_datetime(df['dt'], format='mixed', utc=True)
        return df
    
    @wrap_session
    def delete_duplicates(
        self, 
        ticker: str, 
        interval: T.Trade.INTERVAL, 
        **session_kwargs
    ) -> None:
        ticker_interval = self.get_ticker_interval_id(ticker=ticker, interval=interval, **session_kwargs)
        if ticker_interval is None:
            self.prtinfo(f'DELETE_DUPLICATES -  No data for {ticker} {interval}')
            return
        interval_id = ticker_interval[-1]
        all_candles_df = self.qdf(f"SELECT * FROM $.candle WHERE interval_id = '{interval_id}'", show_head=False, **session_kwargs)
        if all_candles_df is None:
            self.prtinfo(f'DELETE_DUPLICATES -  No candles for {ticker} {interval}')
            return
        
        # Convert the dt column to a datetime object
        if all_candles_df is not None:
            all_candles_df['dt'] = pd.to_datetime(all_candles_df['dt'], format='mixed', utc=True)
        
        duplicates_df = all_candles_df.loc[all_candles_df.duplicated(subset='dt', keep='first')]
        if not duplicates_df.empty:
            delete_ids = duplicates_df['candle_id'].tolist()
            self.prtinfo(f'DELETE_DUPLICATES - Deleting {len(delete_ids)} duplicates for {ticker} {interval}')
            self.delete_fast(table=self.t.candle, ids=delete_ids, **session_kwargs)
        else:
            self.prtinfo(f'DELETE_DUPLICATES - No duplicates for {ticker} {interval}')
    
    @wrap_session
    def update_data(self, ticker: str, interval: T.Trade.INTERVAL, **session_kwargs) -> None:
        #########################################################################################
        def run_delete_duplicates() -> None:            
            # Clean the data by removing any duplcicates
            self.delete_duplicates(ticker=ticker, interval=interval, **session_kwargs)
            session_kwargs['session'].commit()
        #########################################################################################
        ticker = ticker.upper()
        if ticker in self._TICKER_PAIRS:
            y_ticker = self._TICKER_PAIRS[ticker]
        else:
            y_ticker = ticker
        ticker_interval = self.get_ticker_interval_id(ticker=ticker, interval=interval, **session_kwargs)
        most_recent_candle_dt = None
        if ticker_interval is None:
            orm_ticker = self.t.ticker(symbol=ticker)
            self.add(orm_ticker, **session_kwargs)
            orm_interval = self.t.interval(ticker=orm_ticker, interval=interval)
            self.add(orm_interval, **session_kwargs)
            start = self.patch_start_from_interval(interval=interval)
            interval_id = orm_interval.interval_id
        else:            
            interval_id = ticker_interval[1]
            most_recent_candle_dt = self.get_recent_dt(ticker=ticker, interval=interval, **session_kwargs)
            if most_recent_candle_dt is not None:
                next_interval_dt = get_next_interval_dt(interval=interval, dt=most_recent_candle_dt)
                dt_now_utc = datetime.now(tz=timezone.utc)
                if dt_now_utc <= next_interval_dt:
                    self.prtinfo(f'UPDATE - No new data for {ticker} {interval}')
                    run_delete_duplicates()
                    return
                start = most_recent_candle_dt
            else:
                start = self.patch_start_from_interval(interval=interval)

        data = yf.download(y_ticker, start=start, end=None, interval=interval)
        if data.empty:
            self.prtinfo(f'UPDATE - No new Yahoo data for {ticker} {interval} from start: {start.strftime("%Y-%m-%d %H:%M:%S")}')
            run_delete_duplicates()
            return
        
        recent_yahoo = data.index[-1]        
        if most_recent_candle_dt is not None:
            # Subset data that is greater than the most recent candle
            data = data[data.index > most_recent_candle_dt]
        if data.empty:
            _msg = f'UPDATE - No new data compared for {ticker} {interval} for'
            _msg += f' greater than most recent: {most_recent_candle_dt.strftime("%Y-%m-%d %H:%M:%S")}'
            _msg += f' (last Yahoo: {recent_yahoo.strftime("%Y-%m-%d %H:%M:%S")})'
            self.prtinfo(_msg)
            run_delete_duplicates()
            return
        
        str_start = start.strftime('%Y-%m-%d %H:%M:%S')
        str_end = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        self.prtinfo(f'UPDATE - Updating {ticker} {interval} data from {str_start} to {str_end} ({len(data)} records)')
        for dt, row in data.iterrows():
            params = {
                'candle_id': generate_uuid(),
                'interval_id': interval_id,
                'dt': datetime_parse(dt),
                'open': row['Open'].values[0],
                'high': row['High'].values[0],
                'low': row['Low'].values[0],
                'close': row['Close'].values[0],
                'volume': row['Volume'].values[0],
            }
            self.qdf(
                sql="""
                INSERT INTO $.candle (candle_id, interval_id, dt, open, high, low, close, volume) 
                VALUES (:candle_id, :interval_id, :dt, :open, :high, :low, :close, :volume)
                """, 
                parameters=params,
                show_head=False,
                warn_is_none=False, 
                **session_kwargs
            )
        session_kwargs['session'].commit()
        run_delete_duplicates()
    
    def get_recent_dt(self, ticker: str, interval: T.Trade.INTERVAL, **session_kwargs) -> Optional[datetime]:
        ticker_interval = self.get_ticker_interval_id(ticker=ticker, interval=interval, **session_kwargs)
        if ticker_interval is None:
            return None
        interval_id = ticker_interval[1]
        most_recent_candle_df = self.qdf(
            sql=f"""
            SELECT 
                MAX(dt) AS dt 
            FROM 
                candle 
            WHERE 
                interval_id = '{interval_id}'
            """, 
            show_head=False,
            **session_kwargs
        )
        if most_recent_candle_df is not None:
            max_dt = datetime_parse(most_recent_candle_df['dt'].values[0])
            max_dt_utc = max_dt.replace(tzinfo=timezone.utc)
            return max_dt_utc
        else:
            return None
    
    @staticmethod
    def patch_start_from_interval(interval: T.Trade.INTERVAL, dt: Optional[datetime] = None) -> datetime:
        if dt is None:
            dt = datetime.now(tz=timezone.utc)
        if interval == "1m":
            start = dt - timedelta(days=7 - 1)
        elif interval in ("5m", "15m", "30m", "90m"):
            start = dt - timedelta(days=60 - 1)
        elif interval in ("1h", '60m'):
            start = dt - timedelta(days=730 - 1)
        else:
            start = dt - timedelta(days=3650 - 1)
        return start
        

    # Private Methods ################################################################################################################################
    def _at_exit_cleanup(self) -> None:
        """
        This method is registered with atexit and will be called when the program exits. This method will close the engine and dispose of it.
        """
        if self.is_closed:
            return
        else:
            self.close()
    
    def _base_query_load_method(
            self, 
            entity: Type[DeclarativeBase], 
            load_method: Callable, 
            default_load_for: Optional[list] = None, 
            **session_kwargs
        ) -> Query:
        """
        This method will be called by a couple of the query methods above to modify the loading relationships of the query.
        The default load method for our schemas is "selectin" which is generally a good balance between performance and memory usage.
        But it can be very slow with schemas that have many relationships, or have very nested relationships.
        For example the RMU schema has 7 relationships, and for example its child sample stand nests down 6 levels.

        The need for this arose from doing batch updates in the app, batch updates for the Avery property (393 RMUs) took over 10 minutes
        to complete with single RMU querying, took 1.5 minutes to complete with batch RMU querying, but only takes around 1 second for 
        query_lazy() batch querying.

        Currently (as of 06/07/2024) only the query_lazy() method is being used but this method is here in the case that future 
        use-cases need to use a different loading methods.
        """
        relationship_names = entity.get_relationships()  # get_relationships() is a custom method of the DecBaseMixin class (declarative_bases.py)
        if default_load_for is not None:
            for relationship in default_load_for:
                if isinstance(relationship, str):
                    if hasattr(entity, relationship):
                        relationship_names.remove(relationship)
                elif relationship.__class__.__name__ == 'InstrumentedAttribute':
                    relationship_names.remove(relationship.property.key)
        loads = [load_method(getattr(entity, relationship_name)) for relationship_name in relationship_names]
        return self.query(entity, **session_kwargs).options(*loads)

    def _check_transaction(self, session_kwargs: dict) -> Optional[Transaction]:
        trx = None
        # If transaction is not already passed start a new one
        # Note that if there is already a transaction, the return value of this method will be None
        if session_kwargs.get('transaction', None) is None:
            trx = self.start_transaction(
                rollback_on_error=session_kwargs.get('rollback_on_error', True),
            )
            session_kwargs['transaction'] = trx
        return trx

    @staticmethod
    def _convert_column_type_to_pandas_dtype(column_type: str) -> Any:
        """
        This method will take a column_type string and return the corresponding pandas dtype
        :param column_type:     The column_type string
        :return:    Any - The corresponding pandas dtype
        """
        column_type = str(column_type)
        column_type = column_type.upper()
        if startswith(value=column_type, startswith_check=['TEXT', 'STRING', 'VARCHAR']):
            return str
        elif startswith(value=column_type, startswith_check=['REAL', 'FLOAT', 'INTEGER', 'BIGINT', 'SMALLINT', 'INT']):
            return float
        elif startswith(value=column_type, startswith_check=['BOOLEAN', 'BOOL']):
            return bool
        elif startswith(value=column_type, startswith_check=['GEOMETRY', 'BYTES', 'BLOB', 'LARGEBINARY', 'BYTESARRAY']):
            return bytes
        else:
            return object

    def _correct_table_name(self, table_name: str) -> str:
        """
        This method will correct the case of a table_name if it is not found in the database. This method will raise an Exception
        if the table could not be found.
        :param table_name:  The table_name to be corrected
        :return:    str - The corrected table_name
        """
        table_name_lower_dict = {table_name.lower(): table_name for table_name in self.table_names}
        table_name_low = table_name.lower()
        if table_name_low not in table_name_lower_dict:
            raise self.TableNameNotFound(table_name=table_name, current_table_names=list(table_name_lower_dict.values()))
        return table_name_lower_dict[table_name_low]

    @wrap_session
    def _dynamic_table_updater(self, **session_kwargs) -> None:
        """
        This method will look for new tables and/or columns defined from the DeclarativeBase schema classes
        and add them to the database.
        """
        # If __BASE__, then check for any new DeclarativeBase tables and add them automatically
        # If the table does already exist, check for any new columns and add them to the table
        if self.__BASE__ is not None:
            session = session_kwargs['session']
            get_sql_table_name = lambda table_name: f'{self.schema}.{table_name}' if self.schema is not None else table_name

            tables_to_add = []
            is_first_update = True

            # Add new tables first
            for table_name in self.__BASE__.metadata.tables.keys():
                table_object = self.__BASE__.metadata.tables[table_name]
                if table_name not in self.table_names:
                    # Table is new, so add the DeclarativeBase.__table__ to tables_to_add list
                    # DeclarativeBase.metadata.create_all() can take a keyword argument "tables" in which you can target
                    # specific tables for creation
                    tables_to_add.append(table_object)

            if tables_to_add:
                for table in tables_to_add:
                    if is_first_update:
                        is_first_update = False
                        self.prtinfo(text='Detected updates to database schema, making updates...', add_newline=False)
                    self.prtinfo(text=f'Adding new table "{table.name}" to database schema.', add_newline=False)
                self.__BASE__.metadata.create_all(bind=self.engine, tables=tables_to_add)
                
                # Reset the table_names list
                inspector = sa_inspect(self.engine)
                if self.schema is not None:
                    self.table_names = [i for i in inspector.get_table_names(schema=self.schema) if i not in self._IGNORE_DEFAULT_TABLES]
                else:
                    self.table_names = [i for i in inspector.get_table_names() if i not in self._IGNORE_DEFAULT_TABLES]

            # Add new columns to existing tables if any
            for table_name in self.__BASE__.metadata.tables.keys():
                table_object = self.__BASE__.metadata.tables[table_name]
                sql_table_name = get_sql_table_name(table_name)
                # Check for new columns within existing tables
                existing_columns = self.columns(table_name=table_name, return_names_only=True)
                table_columns = list(table_object.columns)
                for column in table_columns:
                    if column.name not in existing_columns:
                        # _prepare_column_sql() uses the SQLAlchemy DDLCompiler class to create
                        # a sql statement similar to what would be generated with CREATE TABLE.
                        # This statement will be paired with an ALTER TABLE statement
                        if is_first_update:
                            is_first_update = False
                            self.prtinfo(text='Detected updates to database schema, making updates...', add_newline=False)
                        self.prtinfo(text=f'Adding column "{column.name}" to existing table "{table_name}"', add_newline=False)
                        column_sql = self._prepare_column_sql(column=column)
                        sql = f"ALTER TABLE {sql_table_name} ADD COLUMN {column_sql};"
                        if self.database_type in ['postgres']:
                            sql += ' COMMIT;'  # Postgres needs commit
                        sql = text(sql)
                        session.execute(sql)
                
                # Check for columns that may have been removed from the table's schema definition, and remove them from the database table
                table_column_names = [col.name for col in table_columns]
                for column_name in existing_columns:
                    if column_name not in table_column_names:
                        if is_first_update:
                            is_first_update = False
                            self.prtinfo(text='Detected updates to database schema, making updates...', add_newline=False)
                        self.prtinfo(text=f'Deleting column "{column_name}" from existing table "{table_name}"', add_newline=False)
                        sql = f"ALTER TABLE {sql_table_name} DROP COLUMN {column_name}"
                        if self.database_type not in ['sqlite', 'access']:
                            sql += ' CASCADE'
                        sql += ';'
                        if self.database_type in ['postgres']:
                            sql += ' COMMIT;'  # Postgres needs commit
                        sql = text(sql)
                        session.execute(sql)

    def _prepare_column_sql(self, column: Union[Column, MappedColumn]) -> str:
        """
        This method will take a SQLAlchemy Column object and return a SQL statement that can be used to add the column to an existing table.
        :param column:  The SQLAlchemy Column object
        :return:    str - The SQL statement that can be used to add the column to an existing table
        """
        table_name = column.table.name
        c = CreateColumn(column)
        ddl = DDLCompiler(dialect=self.engine.dialect, statement=None, schema_translate_map={None: self.schema})
        sql = ddl.visit_create_column(c, first_pk=column.primary_key)
        # Check if column is primary key
        if column.primary_key:
            sql += ' PRIMARY KEY'
        # Check if column has default value
        if column.default is not None:
            default_arg = column.default.arg
            if callable(default_arg):
                default_arg = default_arg(ctx=None)  # Not sure why but "ctx" is necessary
            sql += ' DEFAULT ' + prepare_stringify_value(default_arg, bool_is_int=False)
        else:
            if not column.nullable:
                # Required columns cannot be added where a table has records without a default
                # So first check if the table has any records and if so then look for defaults
                # from constraint values or finally use the default value for the column type
                records_df = self.qdf(f'SELECT * FROM $.{table_name}', show_head=False, warn_is_none=True)
                if records_df is not None:
                    constraints = self.find_column_constraints(column=column)
                    default_found = False
                    if constraints is not None:
                        if isinstance(constraints, tuple):
                            sql += ' DEFAULT ' + prepare_stringify_value(constraints[0], bool_is_int=False)
                            default_found = True
                        else:  # isinstance of dict
                            for controller_column_name in constraints:
                                if column.name in constraints[controller_column_name]:
                                    values_buckets_list = list(constraints[controller_column_name][column.name].values())
                                    if 'other' in values_buckets_list[0]:
                                        sql += " DEFAULT 'other'"
                                        default_found = True
                                        break
                                    else:
                                        sql += ' DEFAULT ' + prepare_stringify_value(values_buckets_list[0][0], bool_is_int=False)
                                        default_found = True
                                        break  
                    if not default_found:
                        col_type_str = str(column.type)
                        for col_type, default_func in self.SQLALCHEMY_TYPE_TO_DEFAULT.items():
                            if col_type_str.startswith(col_type):
                                sql += ' DEFAULT ' + prepare_stringify_value(default_func(), bool_is_int=False)
                                break                 

        # Check if the column has foreign keys
        if column.foreign_keys and self.database_type != 'sqlite':
            # SQLite does not support ADD CONSTRINT with ALTER TABLE
            # Possible todo is to make a temporary table, copy the data, drop the original table, 
            # create the new table with the foreign key, copy the data back            
            for fkey in column.foreign_keys:
                related_table_name, related_field = str(fkey.column).split('.')
                if self.schema is not None:
                    related_table_name = f'{self.schema}.{related_table_name}'
                fkey_sql = f', ADD CONSTRAINT fk_{table_name}_{column.name} FOREIGN KEY({column.name}) REFERENCES {related_table_name}({related_field})'
                if fkey.ondelete:
                    fkey_sql += f' ON DELETE {fkey.ondelete}'
                if fkey.onupdate:
                    fkey_sql += f' ON UPDATE {fkey.onupdate}'
                sql += fkey_sql
        return sql

    # Public Methods #################################################################################################################################
    def close(self) -> None:
        """
        Closes the Session (which is self) and disposes of the engine if present
        :return: None
        """
        # super().close()
        active_transaction_ids = list(self.active_transactions.keys())
        for transaction_id in active_transaction_ids:
            self.active_transactions[transaction_id].close()

        if self.engine is not None:
            self.engine.dispose()
        self.is_closed = True

    def columns(self, table_name: str, return_names_only: bool = False) -> Union[List[str], List[dict]]:
        """
        Returns a list of column names for a given table_name. If return_names_only is True, then only the column names will be returned,
        otherwise a list of dictionaries containing column information will be returned. This method will raise an Exception if the table_name
        is not found.
        :param table_name:          The table_name for which to return column information
        :param return_names_only:   Boolean indicating if only the column names should be returned
        :return:    List[str] or List[dict] - A list of column names or a list of dictionaries containing column information
        """
        table_name = self._correct_table_name(table_name=table_name)
        inspector = sa_inspect(self.engine)
        column_info = inspector.get_columns(table_name=table_name, schema=self.schema)
        if return_names_only:
            return [item['name'] for item in column_info]
        else:
            return column_info

    def copy(self, engine_echo: bool = False, show_description: bool = False, **connection_kwargs):
        """
        Creates a copy of the current DatabaseEngine instance. This method will copy all of the attributes of the current
        :param engine_echo:         The echo parameter for the new engine
        :param show_description:    The show_description parameter for the new engine
        :param connection_kwargs:   Any additional connection kwargs to be passed to the new engine
        :return:    DatabaseEngine - A new DatabaseEngine instance
        """
        kwargs = filter_callable_kwargs(
            func=self.__init__,
            passed_kwargs=self.__dict__,
            remove_predefined_kwargs=['engine_echo', 'show_description']
        )
        new_engine = self.__class__(
            engine_echo=engine_echo,
            show_description=show_description,
            **kwargs,
            **connection_kwargs
        )
        return new_engine
    
    def copy_database(self, copy_to_engine: 'Engine') -> None:        
        # Because of the inheriting class structure of our database schema engines
        # there may be two ways to check if the two engines are compatible for copying the class name and the parent class name.
        # They must be of the same "first: subclass of the master DatabaseEngine class.
        # For example the MBGPostgres and MBGSQLite classes are compatible because they both inherit from
        # the MBGDatabaseEngine class. But the MBGPostgres and BKSqlite classes are not compatible.
        self_name = self.__class__.__name__
        copy_to_engine_name = copy_to_engine.__class__.__name__
        self_parent = self.__class__.__base__.__name__
        copy_to_engine_parent = copy_to_engine.__class__.__base__.__name__
        proceed = False
        if self_name == copy_to_engine_name:
            proceed = True
        elif self_parent == copy_to_engine_name:
            proceed = True
        elif self_name == copy_to_engine_parent:
            proceed = True
        elif self_parent == copy_to_engine_parent:
            proceed = True

        if not proceed:
            raise self.CannotCopyDatabaseError(current_engine_name=self_parent, copy_to_engine_name=copy_to_engine_parent)

        current_trx = self.start_transaction()
        copy_trx = copy_to_engine.start_transaction()
        json_columns_by_table = self.json_columns_all_tables()
        try:
            possible_parents = {}
            for table_name in self.t.hierarchical_table_names(association_tables_last=True):
                if table_name in json_columns_by_table:
                    json_columns = json_columns_by_table[table_name]
                else:
                    json_columns = None

                df = self.qdf(
                    sql=f'SELECT * FROM $.{table_name}', 
                    show_head=False, 
                    json_columns_and_types=json_columns,
                    transaction=current_trx
                )
                if df is not None:
                    if table_name == 'merch':
                        df['species'] = df['species'].apply(lambda x: 'all' if x in ['["all"]', "['all']"] else x)

                    df_dict = df.to_dict(orient='records')
                    if table_name.startswith('table_'):
                        for item in df_dict:
                            sql = f"INSERT INTO $.{table_name} ({', '.join(item.keys())}) VALUES ({', '.join([f':{key}' for key in item.keys()])})"
                            copy_to_engine.qdf(
                                sql=sql, 
                                parameters=item, 
                                show_head=False, 
                                warn_is_none=False, 
                                transaction=copy_trx
                            )
                        copy_trx.commit()

                    else:
                        for item in df_dict:
                            other_objects = {}
                            for key in item:
                                if key.endswith('_id') and not isna(item[key]):
                                    other_table_name = key[:-3]
                                    if other_table_name != table_name:
                                        if other_table_name in possible_parents:
                                            if item[key] in possible_parents[other_table_name]:
                                                other_objects[other_table_name] = possible_parents[other_table_name][item[key]]

                            filtered_kwargs = filter_callable_kwargs(
                                func=copy_to_engine.t[table_name].__init__,
                                passed_kwargs=item
                            )
                            new_obj = copy_to_engine.t[table_name](
                                **filtered_kwargs,
                                **other_objects
                            )
                            # Add other attributes that may not be in __init__ args
                            # such as "harvest_trees" in activity
                            for attribute in item:
                                if not attribute.endswith('_id'):
                                    if attribute not in filtered_kwargs:
                                        if hasattr(new_obj, attribute) and item[attribute] is not None and getattr(new_obj, attribute) is None:
                                            setattr(new_obj, attribute, item[attribute])

                            setattr(new_obj, f'{table_name}_id', item[f'{table_name}_id'])
                            copy_to_engine.add(new_obj, transaction=copy_trx)

                        all_objects = copy_to_engine.query(copy_to_engine.t[table_name], transaction=copy_trx).all()
                        possible_parents[table_name] = {}
                        for obj in all_objects:
                            possible_parents[table_name][getattr(obj, f'{table_name}_id')] = obj
                    print(f'Transferred {len(df)} records from {table_name} to new database.\n')
                else:
                    print(f'No records found in {table_name} to transfer.\n')
            copy_trx.commit()

        finally:
            current_trx.close()
            copy_trx.close()

    def create_all_tables_from_base(self):
        """
        Creates all tables from the DeclarativeBase schema classes
        """
        if self.__BASE__ is None:
            raise self.DeclarativeBaseError(method_name='create_all_tables_from_base')
        self.prtinfo(text='Creating all tables...')
        self.__BASE__.metadata.create_all(bind=self.engine)

    @staticmethod
    def decrypt_value_static(fernet_encryption_key: bytes, value: bytes, original_encoding: str = 'utf-8', ttl: Optional[int] = None) -> str:
        """
        This static method can be used directly if the user has an encryption key but would not like to start a database engine

        :param fernet_encryption_key: The encryption key
        :param value: The value to be decrypted
        :param original_encoding: The original encoding of the value
        :param ttl: The timeout of the key
        :return: str - The string representation of the decrypted value
        """
        fernet = Fernet(fernet_encryption_key)
        return fernet.decrypt(token=value, ttl=ttl).decode(encoding=original_encoding)

    def decrypt_value(self, value: bytes, original_encoding: str = 'utf-8', ttl: Optional[int] = None) -> Any:
        """
        Users can pass a Fernet Encryption Key to the constructor of this class, possibly in the case that fields within
        the database are encrypted. Using this key, this method will decrypt a value. If no key is passed in the constructor
        (or not set after the fact) this method will raise an Exception.

        :param value: The value to be decrypted
        :param original_encoding: The original encoding of the value
        :param ttl: The timeout of the key
        :return: str - The string representation of the decrypted value
        """
        fernet = Fernet(self.fernet_encryption_key)
        return fernet.decrypt(token=value, ttl=ttl).decode(encoding=original_encoding)

    def drop_all_tables_from_base(self):
        """
        Drops all tables from the DeclarativeBase schema classes
        """
        if self.__BASE__ is None:
            raise self.DeclarativeBaseError(method_name='drop_all_tables_from_base')
        self.prtinfo(text='Dropping all tables...')
        self.__BASE__.metadata.drop_all(bind=self.engine)

    @staticmethod
    def dt_now() -> datetime.now:
        """
        Shorthand for returning a datetime.datetime.now() object
        :return: datetime.datetime.now()
        """
        return datetime.now()

    @staticmethod
    def dt_then(
            days: int = 0,
            seconds: int = 0,
            microseconds: int = 0,
            milliseconds: int = 0,
            minutes: int = 0,
            hours: int = 0,
            weeks: int = 0
    ) -> datetime:
        """
        Shorthand for returning a datetime.datetime.now() object + a timedelta
        :return: datetime.datetime
        """
        td = timedelta(
            days=days,
            seconds=seconds,
            microseconds=microseconds,
            milliseconds=milliseconds,
            minutes=minutes,
            hours=hours,
            weeks=weeks
        )
        return datetime.now() + td

    @staticmethod
    def dt_today() -> date.today:
        """
        Shorthand for returning a datetime.date.today() object
        :return: datetime.date.today()
        """
        return date.today()

    def encrypt_value(self, value: Any, encoding: str = 'utf-8') -> bytes:
        """
        Users can pass a Fernet Encryption Key to the constructor of this class, possibly in the case that fields within
        the database are encrypted. Using this key, this method will encrypt a value. If no key is passed in the constructor
        (or not set after the fact) this method will raise an Exception.

        :param value: The value to be encrypted, note that this value will be turned into a string and then bytes before encryption
        :param encoding: The encoding of the value to bytes
        :return: bytes - The encrypted value
        """
        value = bytes(str(value), encoding=encoding)
        fernet = Fernet(self.fernet_encryption_key)
        return fernet.encrypt(data=value)

    def make_session(
            self,
            expire_on_commit: bool = True,
            autobegin: bool = True,
            twophase: bool = False,
            enable_baked_queries: bool = True,
            info: Optional[dict] = None
    ) -> Session:
        """
        Creates a new SQLAlchemy Session object, and binds it with the DatabaseEngine's engine
        The arguments of this method are specific to the SQLAlchemy Session object initializer
        :param autoflush:               Boolean indicating if autoflush should be enabled
        :param expire_on_commit:        Boolean indicating if expire_on_commit should be enabled
        :param autobegin:               Boolean indicating if autobegin should be enabled
        :param twophase:                Boolean indicating if twophase should be enabled
        :param enable_baked_queries:    Boolean indicating if enable_baked_queries should be enabled
        :param info:                    Optional info dictionary
        :return: SQLAlchemy Session object
        """
        return Session(
            bind=self.engine,
            expire_on_commit=expire_on_commit,
            autobegin=autobegin,
            twophase=twophase,
            enable_baked_queries=enable_baked_queries,
            info=info
        )

    @classmethod
    def primary_key_field(cls, table: DeclarativeBase) -> Union[str, list]:
        """
        This method will return the primary key field of a schema class
        :param table: The schema class - typically comes from the SchemaClassHolder (class attribute "t"), i.e mbg_engine.t.rmu
        :return: str|list - The primary key field(s)
        """
        primary_keys = [column.name for column in table.__table__.primary_key.columns]
        if len(primary_keys) == 1:
            return primary_keys[0]
        else:
            return primary_keys

    @classmethod
    def prtinfo(cls, text: str, add_newline: bool = True) -> None:
        """
        Prints a blue colored message to the console with the class name as a prefix
        :param text:            The text to print
        :param add_newline:     Boolean indicating if a newline should be added after the text
        """
        prtcolor(text=text, color_code=34, prefix=f'[{cls.__name__} INFO]', add_prefix=True, add_newline=add_newline)

    @classmethod
    def prtwarn(cls, text: str, add_newline: bool = True) -> None:
        """
        Prints a yellow colored message to the console with the class name as a prefix
        :param text:            The text to print
        :param add_newline:     Boolean indicating if a newline should be added after the text
        """
        prtcolor(text=text, color_code=33, prefix=f'[{cls.__name__} WARNING]', add_prefix=True, add_newline=add_newline)

    def sql_table_name(self, table_name: str) -> str:
        """
        Returns the table_name with the schema prepended if the schema is not None
        :param table_name:  The table_name to be corrected
        :return:    str - The corrected table_name
        """
        table_name = self._correct_table_name(table_name=table_name)
        if self.schema is not None:
            return f'{self.schema}.{table_name}'
        else:
            return table_name

    def start_transaction(
            self,
            hold_commit_until_close: bool = True,
            rollback_on_error: bool = True,

            # Session specific kwargs
            expire_on_commit: bool = False,
            autobegin: bool = True,
            twophase: bool = False,
            enable_baked_queries: bool = True,
            info: Optional[dict] = None,
            user_id: Optional[str] = None
    ) -> Transaction:
        """
        Creates a DatabaseEngine.Transaction object which can be used to execute multiple queries within a transaction
        """
        return self.Transaction(
            engine=self,
            hold_commit_until_close=hold_commit_until_close,
            rollback_on_error=rollback_on_error,
            expire_on_commit=expire_on_commit,
            autobegin=autobegin,
            twophase=twophase,
            enable_baked_queries=enable_baked_queries,
            info=info,
            user_id=user_id
        )

    # Properties #####################################################################################################################################
    @property
    def table_columns(self) -> dict:
        """
        Returns a dictionary of table names and their associated columns
        :return:    dict - {table_name: [column_name, column_name, ...]}
        """
        return {table_name: self.columns(table_name=table_name, return_names_only=True) for table_name in self.table_names}