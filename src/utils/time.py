"""Zaman ve tarih yardımcıları.

Örnek:
    from src.utils.time import utc_timestamp

    now = utc_timestamp()
"""

from __future__ import annotations

import datetime as dt


def utc_timestamp() -> int:
    """Şu anki UTC zaman damgasını saniye cinsinden döndür."""

    return int(dt.datetime.now(tz=dt.timezone.utc).timestamp())


def ensure_utc(ts: dt.datetime) -> dt.datetime:
    """Datetime değerini UTC'ye dönüştür."""

    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)
