from peewee import Model, IntegerField, CharField, FloatField
from . import db


class Symbol(Model):
    id = IntegerField(primary_key=True)
    ticker = CharField()
    name = CharField(null=True)
    market = CharField()
    currency = CharField()

    class Meta:
        database = db
        table_name = "symbol"


class OHLCVBase(Model):
    symbol_id = IntegerField()
    timestamp = IntegerField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

    class Meta:
        database = db


def get_ohlcv_model(bar: str):
    class Meta:
        database = db
        table_name = f"ohlcv_{bar}"

    return type(f"OHLCV{bar}", (OHLCVBase,), {"Meta": Meta})
