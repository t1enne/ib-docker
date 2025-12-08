from zipline.data.bundles import register
from zipline.data.bundles.core import from_dataframe
from src.utils import read_candles


@register('ibkr_bundle')
def ibkr_bundle(symbols):
    for symbol in symbols:
        df = read_candles(symbol)
        # Ensure df has the right format: index datetime, columns open high low close volume
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index.name = 'date'
        yield from from_dataframe(df, symbol=symbol)</content>
<parameter name="filePath">src/bt/bundles.py