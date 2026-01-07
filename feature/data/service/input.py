import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datetime import datetime
import pandas as pd
from config import COIN, DATA


class InputService:
    def run(
        self,
        symbol: str = None,
        timeframe: str = None,
        since: datetime = None,
        until: datetime = None,
    ) -> pd.DataFrame:
        symbol = symbol or COIN.SYMBOL
        timeframe = timeframe or COIN.TIMEFRAME[0]
        since = since or COIN.START_DATE
        until = until or COIN.END_DATE
        name = DATA.INPUT

        match name:
            case "ccxt":
                from feature.data.input.ccxt import CCXTInput
                return CCXTInput().run()
            # case "csv":
            #     from feature.data.input.csv import CSVInput
            #     return CSVInput().run()
            case _:
                raise ValueError(f"Unsupported input source: {name}")
