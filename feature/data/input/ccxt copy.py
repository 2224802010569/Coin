import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ccxt
from datetime import datetime
from config import COIN
from feature.data.entities.candle import Candle
from pyspark.sql import DataFrame
from feature.data.service.spark import SparkService

class CCXTInput:
    def run(
        self,
        symbol: str = None,
        timeframe: str = None,
        since: datetime = None,
        until: datetime = None,
    ) -> DataFrame:
        
        symbol = symbol or COIN.SYMBOL
        timeframe = timeframe or COIN.TIMEFRAME[0]
        since = since or COIN.START_DATE
        until = until or COIN.END_DATE

        exchange_cls = getattr(ccxt, COIN.EXCHANGE)
        exchange = exchange_cls({
            "enableRateLimit": True,
            "timeout": int(10 * 1000),
        })

        since_ms = int(since.timestamp() * 1000)
        Candles = []

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit=1000)
            if not ohlcv:
                break
            for c in ohlcv:
                ts = datetime.utcfromtimestamp(c[0] / 1000)
                if ts > until:
                    break
                Candles.append((
                    ts,
                    float(c[1]),  # open
                    float(c[2]),  # high
                    float(c[3]),  # low
                    float(c[4]),  # close
                    float(c[5]),  # volume
                    timeframe
                ))
            if len(ohlcv) < 1000 or ts >= until:
                break
            since_ms = int(ohlcv[-1][0]) + 1
            spark = SparkService.spark()
            schema = SparkService.schema_from_entity(Candle)
        return spark.createDataFrame(Candles, schema)