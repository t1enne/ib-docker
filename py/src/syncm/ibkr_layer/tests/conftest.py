import os
import pytest
from peewee import SqliteDatabase
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

# Test database path
TEST_DB_PATH = os.path.join(os.getcwd(), "..", "data", "test_db.sqlite")


@pytest.fixture(scope="session")
def test_db():
    """Create test database with required tables."""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(TEST_DB_PATH), exist_ok=True)

    # Remove existing test db to start fresh
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    db = SqliteDatabase(TEST_DB_PATH, pragmas={"journal_mode": "wal"})

    # Create tables
    from src.db.models import Symbol, OHLCVBase

    db.bind([Symbol])
    db.create_tables([Symbol])

    # Create dynamic OHLCV tables for different bar sizes
    bar_sizes = ["1h", "1d"]
    for bar in bar_sizes:
        model = get_ohlcv_model_for_test(db, bar)
        db.create_tables([model])

    yield db

    # Cleanup
    db.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture(autouse=True)
def clean_tables(test_db):
    """Clean all tables before each test."""
    from src.db.models import Symbol

    # Delete from symbol table
    Symbol.delete().execute()

    # Delete from all OHLCV tables
    bar_sizes = ["1h", "1d"]
    for bar in bar_sizes:
        model = get_ohlcv_model_for_test(test_db, bar)
        model.delete().execute()

    yield


def get_ohlcv_model_for_test(db_instance, bar: str):
    """Get OHLCV model bound to test database using dynamic class creation."""
    from peewee import Model, IntegerField, FloatField

    table_name_val = f"ohlcv_{bar}"
    class_name = f"OHLCV{bar.upper()}"

    # Create Meta class with bound values
    meta_attrs = {
        "database": db_instance,
        "table_name": table_name_val,
    }
    Meta = type("Meta", (), meta_attrs)

    # Use type() to create class dynamically to avoid closure issues
    attrs = {
        "symbol_id": IntegerField(),
        "timestamp": IntegerField(index=True),
        "open": FloatField(),
        "high": FloatField(),
        "low": FloatField(),
        "close": FloatField(),
        "volume": FloatField(),
        "Meta": Meta,
    }

    return type(class_name, (Model,), attrs)


@pytest.fixture
def mock_symbol():
    """Create a mock symbol for testing."""
    symbol = MagicMock()
    symbol.id = 12345
    symbol.ticker = "AAPL"
    return symbol


@pytest.fixture
def sample_candle_data():
    """Generate sample candle data for testing."""

    def _generate(start_timestamp_ms, num_candles, interval_ms=3600000):
        """Generate candles starting from start_timestamp_ms.

        Args:
            start_timestamp_ms: Start timestamp in milliseconds
            num_candles: Number of candles to generate
            interval_ms: Interval between candles in milliseconds (default 1 hour)
        """
        candles = []
        for i in range(num_candles):
            timestamp = start_timestamp_ms + (i * interval_ms)
            base_price = 100.0 + (i * 0.1)  # Slight price drift
            candles.append(
                {
                    "t": timestamp,
                    "o": base_price,
                    "h": base_price + 1.0,
                    "l": base_price - 1.0,
                    "c": base_price + 0.5,
                    "v": 1000 + i,
                }
            )
        return candles

    return _generate


@pytest.fixture
def mock_get_contract_info(mock_symbol):
    """Mock get_contract_info to return test symbol."""
    with patch(
        "src.syncm.ibkr_layer.candles.get_contract_info", new_callable=AsyncMock
    ) as mock:
        mock.return_value = mock_symbol
        yield mock


@pytest.fixture
def ibkr_api_mock():
    """Mock IBKR API routes using respx for real HTTP client testing."""
    import respx
    from httpx import Response

    with respx.mock(base_url="https://localhost:5000/v1/api/") as mock:
        yield mock


@pytest.fixture(autouse=True)
def patch_test_db(test_db):
    """Patch the production database with test database for candles module."""
    from src.db.models import Symbol, OHLCVBase
    import src.db as db_module

    # Create models bound to test database
    test_models = {}

    def patched_get_ohlcv_model(bar: str):
        """Return OHLCV model bound to test database."""
        if bar in test_models:
            return test_models[bar]

        class Meta:
            database = test_db
            table_name = f"ohlcv_{bar}"

        model = type(f"OHLCV{bar}", (OHLCVBase,), {"Meta": Meta})
        test_models[bar] = model
        return model

    # Bind Symbol model to test database
    test_db.bind([Symbol])

    # Patch the db module's db object
    original_db = db_module.db
    db_module.db = test_db

    # Patch get_ohlcv_model functions
    with patch("src.db.models.get_ohlcv_model", side_effect=patched_get_ohlcv_model):
        with patch(
            "src.syncm.ibkr_layer.candles.get_ohlcv_model",
            side_effect=patched_get_ohlcv_model,
        ):
            yield test_db

    # Restore original db
    db_module.db = original_db
