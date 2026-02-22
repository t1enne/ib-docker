from dataclasses import dataclass
from peewee import Model, IntegerField, CharField, FloatField
from . import db

from dataclasses import dataclass
from typing import Optional


@dataclass
class ISymbol:
    """
    Dataclass matching SymbolSchema Peewee model
    """

    conid: int
    ticker: str
    market: str
    currency: str
    name: Optional[str] = None


@dataclass
class ICandle:
    """
    Dataclass matching CandleSchema Peewee model
    """

    conid: int
    ticker: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class SymbolSchema(Model):
    conid = IntegerField(primary_key=True)
    ticker = CharField()
    name = CharField(null=True)
    market = CharField()
    currency = CharField()

    class Meta:
        database = db
        table_name = "symbol"


class CandleSchema(Model):
    conid = IntegerField()
    ticker = CharField()
    timestamp = IntegerField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

    class Meta:
        database = db
        table_name = "candle"
