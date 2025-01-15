import atexit
import json
import numpy as np
import os
import pandas as pd
import pyodbc
import re
import sqlalchemy
import textwrap
import traceback
import yfinance as yf

from cryptography.fernet import Fernet
from datetime import datetime, date, timedelta, timezone
from functools import wraps
from hashlib import sha256
from sqlalchemy import (
    and_,
    create_engine,
    event,
    func,
    inspect as sa_inspect,
    or_,
    select,
    table,
    text,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Constraint,
    DateTime,
    Engine,
    Integer,
    Float,
    ForeignKey,
    MetaData,
    LargeBinary,
    String,
    Table,
    UniqueConstraint
)
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    mapped_column,
    lazyload,
    joinedload,
    subqueryload,
    noload,
    raiseload,
    selectinload,
    registry,
    relationship,
    sessionmaker,
    DeclarativeBase,
    Mapped,
    Query,
    Session,
    MappedColumn
)
from sqlalchemy.orm.attributes import set_committed_value, InstrumentedAttribute
from sqlalchemy.orm.session import object_session
from sqlalchemy.schema import CreateColumn
from sqlalchemy.sql.compiler import DDLCompiler
from sqlalchemy.sql.schema import CheckConstraint
from sqlalchemy.types import TypeDecorator, UserDefinedType
from threading import Thread
from time import sleep, perf_counter

from ..utils.typing import *
from ..utils.utils import *