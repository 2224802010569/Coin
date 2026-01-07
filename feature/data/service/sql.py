import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sqlite3
from dataclasses import fields
import pandas as pd
from config import DATA


class SQLService:
    def __init__(self, entity):
        self.entity = entity
        self.name = entity.__name__.lower()
        self.storage_dir = DATA.STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / f"data.db"

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def create_table(self):
        with self._connect():
            pass
        cols = []
        for f in fields(self.entity):
            if f.type in (int,):
                sql_type = "INTEGER"
            elif f.type in (float,):
                sql_type = "REAL"
            else:
                sql_type = "TEXT"
            cols.append(f"{f.name} {sql_type}")
        col_sql = ", ".join(cols)
        sql = f"""
        DROP TABLE IF EXISTS {self.name};
        CREATE TABLE {self.name} ({col_sql});
        """
        with self._connect() as conn:
            conn.executescript(sql)

    def read_table(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql(f"SELECT * FROM {self.name}", conn)

    def update_table(self, data: pd.DataFrame = None):
        if data is None or data.empty:
            return
        with self._connect() as conn:
            data.to_sql(
                self.name,
                conn,
                if_exists="append",
                index=False
            )

    def delete_table(self):
        with self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self.name}")
