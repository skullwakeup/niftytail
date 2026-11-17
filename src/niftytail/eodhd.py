"""Automatic end-of-day update from EODHD (D60): the ``fetch-api`` command.

niftyindices.com and nseindia.com forbid automated collection, so the scheduled update asks
EODHD, a licensed vendor, for the NSEI.INDX end-of-day rows (one request, at most one retry,
30 s timeout; standard library only). The rows overlapping the data already loaded (raw, live,
api) must agree exactly: the same session dates and the same closes to 2 decimals; only rows
after the last loaded date are written, to ``data/api/<first>_<last>.csv`` (gitignored), and
``data/api/MANIFEST.toml`` is rewritten. Nothing from this source is committed or published.

The key comes only from ``NIFTYTAIL_EODHD_KEY``. It and the request URL are never logged,
printed, written or put in an error message: failures report the HTTP status or the
exception class only. Logs carry counts only (D23): no date, value or key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from niftytail import live
from niftytail.config import Config

__all__ = ["KEY_ENV", "NO_KEY", "FetchError", "parse", "run"]

URL = "https://eodhd.com/api/eod/NSEI.INDX"
KEY_ENV = "NIFTYTAIL_EODHD_KEY"
NO_KEY = "set NIFTYTAIL_EODHD_KEY"
IST = timezone(timedelta(hours=5, minutes=30), "Asia/Kolkata")  # no DST since 1945
CLOSE = time(16, 30)  # a row dated today is kept only from 16:30 Asia/Kolkata
LOOKBACK_DAYS = 30
TIMEOUT = 30
FIELDS = ("open", "high", "low", "close")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_urlopen: Callable[..., Any] = urllib.request.urlopen  # replaced by a fake in the tests
Row = tuple[date, float, float, float, float]


class FetchError(Exception):
    """A failed or rejected fetch; the message never contains the key, the URL or data."""


def _get(url: str) -> bytes:
    """One GET, retried once on a network failure (not on an HTTP status)."""
    req = urllib.request.Request(url, headers={"User-Agent": "niftytail"})
    err = ""
    for _ in range(2):
        try:
            with _urlopen(req, timeout=TIMEOUT) as resp:
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            err = f"HTTP {exc.code}"
            break
        except Exception as exc:  # only the class name is ever reported
            err = type(exc).__name__
    raise FetchError(f"EODHD request failed: {err}")


def _real(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def parse(blob: bytes) -> list[Row]:
    """The response rows (date, open, high, low, close), checked by the D25 row rules.

    Raises:
        FetchError: not JSON, not a list of rows of the documented shape, a non-finite or
            non-positive price, or dates that are not ascending and unique.
    """
    try:
        data = json.loads(blob)
    except ValueError:
        raise FetchError("EODHD response is not valid JSON") from None
    if not isinstance(data, list):
        raise FetchError("EODHD response: not a list of rows")
    rows: list[Row] = []
    for r in data:
        ok = isinstance(r, dict) and isinstance(r.get("date"), str)
        ok = ok and bool(_ISO.fullmatch(r["date"])) and all(_real(r.get(f)) for f in FIELDS)
        if not ok:
            raise FetchError("EODHD response: a row has the wrong shape")
        try:
            d = date.fromisoformat(r["date"])
        except ValueError:
            raise FetchError("EODHD response: a row has the wrong shape") from None
        rows.append((d, *(float(r[f]) for f in FIELDS)))  # type: ignore[arg-type]
    errors = live.row_errors(rows)
    if errors:
        raise FetchError("EODHD response: " + "; ".join(errors))
    return rows


def _manifest(base: Path, retrieved: date) -> str:
    lines = [f'source = "{live.API_SOURCE}"', f'retrieved = "{retrieved.isoformat()}"']
    for p in sorted((base / live.API_DIR).glob("*.csv")):
        blob = p.read_bytes()
        rows = live.parse_api_csv(blob)
        lines += [
            "",
            "[[file]]",
            f'name = "{p.name}"',
            f'sha256 = "{hashlib.sha256(blob).hexdigest()}"',
            f"rows = {len(rows)}",
            f'first = "{rows[0][0].isoformat()}"',
            f'last = "{rows[-1][0].isoformat()}"',
        ]
    return "\n".join(lines) + "\n"


def run(
    cfg: Config,
    root: str | Path,
    log: logging.Logger,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> int:
    """``fetch-api`` (D60 (d)): exit 0 written or "no new session", 1 on any error.

    Raises:
        FetchError: a failed request or a rejected response (nothing written).
        DataValidationError: from the loaders (nothing written).
    """
    env = os.environ if environ is None else environ
    key = env.get(KEY_ENV, "").strip()
    if not key:
        log.error(NO_KEY)
        return 1
    base = Path(root)
    df, _, _, _ = live.load_series_full(cfg, base)
    last = df.index[-1].date()
    clock = datetime.now(IST) if now is None else now.astimezone(IST)
    today = clock.date()
    start = last - timedelta(days=LOOKBACK_DAYS)
    query = {"api_token": key, "fmt": "json", "from": start.isoformat(), "to": today.isoformat()}
    rows = parse(_get(f"{URL}?{urllib.parse.urlencode(query)}"))
    if any(not start <= r[0] <= today for r in rows):
        raise FetchError("EODHD response: a row lies outside the requested range")
    if rows and rows[-1][0] == today and clock.time() < CLOSE:
        rows = rows[:-1]  # today's close is not final before 16:30 Asia/Kolkata
    loaded = {
        d.date(): float(c)
        for d, c in zip(df.index, df["close"].to_numpy(), strict=True)
        if start <= d.date() <= last
    }
    got = {r[0]: r[4] for r in rows if r[0] <= last}
    if set(got) != set(loaded):
        raise FetchError("EODHD overlap: the session dates differ from the loaded data")
    if any(round(got[d], 2) != round(loaded[d], 2) for d in got):
        raise FetchError("EODHD overlap: a close differs from the loaded data (2 decimals)")
    new = [r for r in rows if r[0] > last]
    counts = {"fetched": len(rows), "overlap_checked": len(got), "new": len(new)}
    if not new:
        log.info("no new session", extra=counts)
        return 0
    out = base / live.API_DIR
    out.mkdir(parents=True, exist_ok=True)
    body = [live.API_HEADER] + [
        ",".join([r[0].isoformat(), *(repr(v) for v in r[1:])]) for r in new
    ]
    (out / f"{new[0][0].isoformat()}_{new[-1][0].isoformat()}.csv").write_bytes(
        ("\n".join(body) + "\n").encode()
    )
    (base / live.API_MANIFEST).write_bytes(_manifest(base, today).encode())
    log.info("fetch-api: written", extra=counts)
    return 0
