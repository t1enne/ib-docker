import os
import signal
import sys
from peewee import SqliteDatabase

db_path = os.path.join(os.getcwd(), "..", "data", "db.sqlite")
db = SqliteDatabase(db_path, pragmas={'journal_mode': 'wal'})

def close_db():
    print("Closing database...")
    db.close()

signal.signal(signal.SIGTERM, lambda signum, frame: (close_db(), sys.exit(0)))
signal.signal(signal.SIGINT, lambda signum, frame: (close_db(), sys.exit(0)))