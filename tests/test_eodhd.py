"""Day-13F fetch-api and the data/api layer (D60). Synthetic data, no network.

urlopen is replaced by a fake; any real socket connection fails the test. The fake key must
never reach stdout, stderr, the logs or a written file.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import socket
import subprocess
import urllib.error
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from conftest import PROJECT_ROOT
from niftytail import data as dl
from niftytail import eodhd, live
from niftytail.__main__ import main
from niftytail.config import Config
from test_live import CONFIG, RAW_END, RETRIEVED, _logged, _nodate, add_live, make_repo

KEY = "FAKE-KEY-0123456789abcdef"
IST = eodhd.IST
NOW = datetime(1996, 7, 2, 17, 0, tzinfo=IST)  # a Tuesday after 16:30


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(eodhd, "_urlopen", refuse)


class _Resp:
    def __init__(self, blob: bytes) -> None:
        self.blob = blob

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_a: object) -> None:
        return None

    def read(self) -> bytes:
        return self.blob


def _fake(monkeypatch: pytest.MonkeyPatch, *answers: Any) -> list[str]:
    """Each call takes the next answer: bytes (a body) or an exception to raise."""
    urls: list[str] = []
    queue = list(answers)

    def urlopen(req: Any, timeout: float) -> _Resp:
        urls.append(req.full_url)
        assert timeout == 30
        a = queue.pop(0)
        if isinstance(a, BaseException):
            raise a
        return _Resp(a)

    monkeypatch.setattr(eodhd, "_urlopen", urlopen)
    return urls


def _rows(cfg: Config, root: Path, start: date) -> list[dict[str, Any]]:
    """The loaded sessions from ``start`` on, as EODHD would return them."""
    df, _ = live.load_series(cfg, root)
    return [
        {
            "date": d.date().isoformat(),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "adjusted_close": float(r["close"]),
            "volume": 0,
        }
        for d, r in df.iterrows()
        if d.date() >= start
    ]


def _new(d: date, close: float = 1234.5) -> dict[str, Any]:
    return {
        "date": d.isoformat(),
        "open": close,
        "high": close + 5,
        "low": close - 5,
        "close": close,
        "adjusted_close": close,
        "volume": 0,
    }


def _body(rows: list[dict[str, Any]]) -> bytes:
    return json.dumps(rows).encode()


def _fetch(cfg: Config, root: Path, now: datetime = NOW, env: dict[str, str] | None = None) -> Any:
    e = {eodhd.KEY_ENV: KEY} if env is None else env
    return _logged(lambda c, r, log: eodhd.run(c, r, log, e, now), cfg, root)


def _files(root: Path) -> bytes:
    return b"".join(
        p.read_bytes() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts
    )


def _start() -> date:
    return RAW_END - timedelta(days=eodhd.LOOKBACK_DAYS)


def test_success_writes_new_rows_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_repo(tmp_path)
    old = _rows(cfg, tmp_path, _start())
    urls = _fake(monkeypatch, _body([*old, _new(date(1996, 7, 1)), _new(date(1996, 7, 2), 1240.0)]))
    status, logs = _fetch(cfg, tmp_path)
    assert status == 0 and _nodate(logs) and len(urls) == 1
    assert logs[-1]["new"] == 2 and logs[-1]["overlap_checked"] == len(old)
    assert "from=1996-05-29" in urls[0] and "to=1996-07-02" in urls[0] and "NSEI.INDX" in urls[0]
    (csv,) = (tmp_path / live.API_DIR).glob("*.csv")
    assert csv.name == "1996-07-01_1996-07-02.csv"
    assert csv.read_text(encoding="utf-8").splitlines()[0] == "date,open,high,low,close"
    man = (tmp_path / live.API_MANIFEST).read_text(encoding="utf-8")
    assert 'source = "EODHD NSEI.INDX"' in man and "rows = 2" in man and KEY not in man
    df, sources, _, api_sha = live.load_series_full(cfg, tmp_path)
    assert sources[-2:] == ["api", "api"] and sources.count("api") == 2
    assert df.index[-1].date() == date(1996, 7, 2) and df["close"].iloc[-1] == 1240.0
    assert api_sha == live.ex.file_sha256(tmp_path / live.API_MANIFEST)
    # the next run sees nothing new
    _fake(monkeypatch, _body(_rows(cfg, tmp_path, date(1996, 6, 2))))
    status, logs = _fetch(cfg, tmp_path, NOW + timedelta(days=1))
    assert status == 0 and logs[-1]["message"] == "no new session" and logs[-1]["new"] == 0
    assert len(list((tmp_path / live.API_DIR).glob("*.csv"))) == 1
    assert KEY.encode() not in _files(tmp_path)


@pytest.mark.parametrize(("hour", "kept"), [(16, False), (17, True)])
def test_todays_row_only_after_1630(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hour: int, kept: bool
) -> None:
    cfg = make_repo(tmp_path)
    now = datetime(1996, 7, 2, hour, 29 if hour == 16 else 0, tzinfo=IST)
    _fake(
        monkeypatch,
        _body([*_rows(cfg, tmp_path, _start()), _new(date(1996, 7, 1)), _new(date(1996, 7, 2))]),
    )
    assert _fetch(cfg, tmp_path, now)[0] == 0
    df, _ = live.load_series(cfg, tmp_path)
    assert (df.index[-1].date() == date(1996, 7, 2)) == kept


def test_clock_is_read_in_kolkata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    utc = datetime(1996, 7, 2, 11, 30, tzinfo=UTC)  # 17:00 in Kolkata
    _fake(monkeypatch, _body([*_rows(cfg, tmp_path, _start()), _new(date(1996, 7, 2))]))
    assert _fetch(cfg, tmp_path, utc)[0] == 0
    assert live.load_series(cfg, tmp_path)[0].index[-1].date() == date(1996, 7, 2)


def test_weekend_row_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    sat = date(1996, 6, 29)  # a Muhurat-style weekend session
    _fake(monkeypatch, _body([*_rows(cfg, tmp_path, _start()), _new(sat), _new(date(1996, 7, 1))]))
    assert _fetch(cfg, tmp_path)[0] == 0
    assert sat in {d.date() for d in live.load_series(cfg, tmp_path)[0].index}


def _rejected(tmp_path: Path, cfg: Config, message: str) -> None:
    out, err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
        pytest.raises(eodhd.FetchError, match=message) as exc,
    ):
        _fetch(cfg, tmp_path)
    assert KEY not in str(exc.value) and "eodhd.com" not in str(exc.value)
    assert KEY not in out.getvalue() + err.getvalue()
    assert not (tmp_path / live.API_DIR).exists()  # nothing written


def test_overlap_date_set_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    old = _rows(cfg, tmp_path, _start())
    _fake(monkeypatch, _body([*old[:3], *old[4:], _new(date(1996, 7, 1))]))
    _rejected(tmp_path, cfg, "session dates differ")


def test_overlap_close_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    old = _rows(cfg, tmp_path, _start())
    old[5]["close"] += 0.01
    _fake(monkeypatch, _body([*old, _new(date(1996, 7, 1))]))
    _rejected(tmp_path, cfg, "a close differs")


def test_overlap_close_rounding_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    old = _rows(cfg, tmp_path, _start())
    old[5]["close"] += 0.001  # equal after rounding to 2 decimals
    _fake(monkeypatch, _body([*old, _new(date(1996, 7, 1))]))
    assert _fetch(cfg, tmp_path)[0] == 0


@pytest.mark.parametrize(
    ("answers", "message", "calls"),
    [
        (
            (
                urllib.error.HTTPError(
                    "https://eodhd.com/x?api_token=" + KEY, 403, "Forbidden", None, None
                ),
            ),
            "HTTP 403",
            1,
        ),  # type: ignore[arg-type]
        ((TimeoutError("timed out"), TimeoutError("timed out")), "TimeoutError", 2),
        ((urllib.error.URLError("no route " + KEY),) * 2, "URLError", 2),
        ((b"{not json",), "not valid JSON", 1),
        ((b'{"date": "1996-07-01"}',), "not a list", 1),
        ((_body([{"date": "1996-07-01", "close": 1.0}]),), "wrong shape", 1),
        (
            (_body([{"date": "1996-02-30", "open": 1, "high": 1, "low": 1, "close": 1}]),),
            "wrong shape",
            1,
        ),
        ((_body([_new(date(1996, 7, 1), -5.0)]),), "finite and positive", 1),
        ((_body([_new(date(1996, 7, 2)), _new(date(1996, 7, 1))]),), "ascending", 1),
        ((_body([_new(date(1996, 7, 9))]),), "outside the requested range", 1),
    ],
)
def test_failures_write_nothing_and_hide_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: Any, message: str, calls: int
) -> None:
    cfg = make_repo(tmp_path)
    urls = _fake(monkeypatch, *answers)
    _rejected(tmp_path, cfg, message)
    assert len(urls) == calls


def test_retry_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    urls = _fake(
        monkeypatch,
        TimeoutError(),
        _body([*_rows(cfg, tmp_path, _start()), _new(date(1996, 7, 1))]),
    )
    assert _fetch(cfg, tmp_path)[0] == 0 and len(urls) == 2


@pytest.mark.parametrize("env", [{}, {eodhd.KEY_ENV: ""}, {eodhd.KEY_ENV: "  "}])
def test_missing_key(tmp_path: Path, env: dict[str, str]) -> None:
    cfg = make_repo(tmp_path)
    status, logs = _fetch(cfg, tmp_path, env=env)
    assert status == 1 and logs[-1]["message"] == "set NIFTYTAIL_EODHD_KEY"
    assert not (tmp_path / live.API_DIR).exists()


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, out.getvalue(), err.getvalue()


def test_cli_never_shows_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(eodhd.KEY_ENV, raising=False)
    status, out, err = _cli("fetch-api", CONFIG)
    assert status == 1 and "set NIFTYTAIL_EODHD_KEY" in err
    assert _cli("fetch-api", "configs/missing.toml")[0] == 1
    monkeypatch.setenv(eodhd.KEY_ENV, KEY)
    _fake(
        monkeypatch,
        urllib.error.HTTPError("https://eodhd.com/?api_token=" + KEY, 401, "no", None, None),
    )  # type: ignore[arg-type]
    status, out, err = _cli("fetch-api", CONFIG)
    assert status == 1 and "HTTP 401" in err and KEY not in out + err
    _fake(monkeypatch, _body([*_rows(cfg, tmp_path, _start()), _new(date(1996, 7, 1))]))
    status, out, err = _cli("fetch-api", CONFIG)
    assert (
        status == 0 and KEY not in out + err and _nodate([json.loads(x) for x in err.splitlines()])
    )
    (tmp_path / live.API_MANIFEST).write_text("broken", encoding="utf-8")
    status, out, err = _cli("fetch-api", CONFIG)
    assert status == 1 and "run fetch-api" in err
    assert KEY.encode() not in _files(tmp_path)


# --- the data/api layer in load_series (D60 (e)) ---------------------------------------------
def _api_file(root: Path, rows: list[tuple[date, float]]) -> None:
    d = root / live.API_DIR
    d.mkdir(parents=True, exist_ok=True)
    body = [live.API_HEADER] + [f"{x.isoformat()},{c!r},{c + 5!r},{c - 5!r},{c!r}" for x, c in rows]
    (d / f"{rows[0][0].isoformat()}_{rows[-1][0].isoformat()}.csv").write_text(
        "\n".join(body) + "\n", encoding="utf-8", newline="\n"
    )
    (root / live.API_MANIFEST).write_bytes(eodhd._manifest(root, RETRIEVED).encode())


def test_precedence_raw_live_api(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    add_live(tmp_path)  # 1996-06-29 .. 1996-08-30
    assert _logged(live.validate_command, cfg, tmp_path, RETRIEVED)[0] == 0
    df, _ = live.load_series(cfg, tmp_path)
    tail = [(d.date(), float(c)) for d, c in df["close"].items()][-3:]
    _api_file(tmp_path, [*tail, (date(1996, 9, 2), 1500.0)])  # overlap with live, then one new
    full, sources, live_sha, api_sha = live.load_series_full(cfg, tmp_path)
    assert len(full) == len(df) + 1 and sources[-1] == "api" and sources.count("api") == 1
    assert "live" in sources and sources[0] == "raw" and live_sha and api_sha
    assert full.index.is_monotonic_increasing and full.index.is_unique
    # the forecast key and the provenance include the api manifest
    a = live.forecast_dir(tmp_path, "c", "r", live_sha, "g")
    assert a != live.forecast_dir(tmp_path, "c", "r", live_sha, "g", api_sha)


@pytest.mark.parametrize("how", ["close", "dates", "manifest", "extra", "header", "row", "empty"])
def test_api_overlap_and_manifest_errors(tmp_path: Path, how: str) -> None:
    cfg = make_repo(tmp_path)
    df, _ = live.load_series(cfg, tmp_path)
    tail = [(d.date(), float(c)) for d, c in df["close"].items()][-3:]
    new = (date(1996, 7, 1), 1500.0)
    if how == "close":
        _api_file(tmp_path, [tail[0], (tail[1][0], tail[1][1] + 0.01), tail[2], new])
        match = "a close differs"
    elif how == "dates":
        _api_file(tmp_path, [tail[0], tail[2], new])
        match = "other dates"
    else:
        _api_file(tmp_path, [new])
        (csv,) = (tmp_path / live.API_DIR).glob("*.csv")
        if how == "manifest":
            csv.write_bytes(csv.read_bytes().replace(b"1500.0,", b"1501.0,", 1))
        elif how == "extra":
            (tmp_path / live.API_DIR / "x.csv").write_bytes(csv.read_bytes())
        elif how == "header":
            csv.write_bytes(b"d,o,h,l,c\n")
            match = "wrong header"
        elif how == "row":
            csv.write_bytes((live.API_HEADER + "\n1996-07-01,1,2\n").encode())
            match = "malformed row"
        else:
            csv.write_bytes((live.API_HEADER + "\n").encode())
            match = "empty"
        if how in ("manifest", "extra"):
            match = "run fetch-api"
    with pytest.raises(dl.DataValidationError, match=match):
        live.load_series(cfg, tmp_path)


def test_no_api_dir_changes_nothing(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    _, sources, live_sha, api_sha = live.load_series_full(cfg, tmp_path)
    assert api_sha is None and live_sha is None and set(sources) == {"raw"}
    (tmp_path / live.API_DIR).mkdir(parents=True)
    with pytest.raises(dl.DataValidationError, match="run fetch-api"):
        live.load_series(cfg, tmp_path)  # a data/api dir without its manifest


def test_forecast_and_series_with_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    _fake(monkeypatch, _body([*_rows(cfg, tmp_path, _start()), _new(date(1996, 7, 1))]))
    assert _fetch(cfg, tmp_path)[0] == 0
    status, _ = _logged(live.run, cfg, CONFIG, 1, False, tmp_path)
    assert status == 0
    (path,) = (tmp_path / live.LIVE_REPORTS).glob("*/forecast.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["provenance"]["api_manifest_sha256"] == live.ex.file_sha256(
        tmp_path / live.API_MANIFEST
    )
    ser = json.loads((path.parent / live.SERIES_JSON).read_text(encoding="utf-8"))
    assert ser["history"][-1]["source"] == "api"


# --- scripts/update.cmd (D60 (f), order per D61 (a)) ------------------------------------------
SCRIPT = PROJECT_ROOT / "scripts" / "update.cmd"


def test_update_cmd_order() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    steps = [
        "call :step validate-live validate-live",
        "call :step fetch-api fetch-api",
        "call :step forecast forecast",
        "call :step check-run forecast --check-run",
    ]
    where = [text.index(s) for s in steps]
    assert where == sorted(where)
    assert text.index('if not exist "data\\live\\" goto fetch') < where[0]


@pytest.mark.skipif(os.name != "nt" or shutil.which("uv") is None, reason="Windows cmd and uv")
def test_update_cmd_prints_the_error_lines(tmp_path: Path) -> None:
    """The real script in a synthetic repo (the project env via UV_PROJECT): validate-live
    passes, fetch-api fails without the key, its error lines print, and nothing else runs."""
    cfg = make_repo(tmp_path)
    add_live(tmp_path)
    assert _logged(live.validate_command, cfg, tmp_path, RETRIEVED)[0] == 0
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "update.cmd")
    env = {k: v for k, v in os.environ.items() if k != eodhd.KEY_ENV}
    env["UV_PROJECT"] = str(PROJECT_ROOT)
    res = subprocess.run(
        ["cmd", "/c", str(tmp_path / "scripts" / "update.cmd")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    lines = res.stdout.splitlines()
    assert res.returncode == 1
    assert lines[:2] == ["validate-live: exit 0", "fetch-api: exit 1"]
    assert any("set NIFTYTAIL_EODHD_KEY" in x for x in lines[2:])
    assert not any(x.startswith(("forecast:", "check-run:")) for x in lines)
    assert not (tmp_path / live.API_DIR).exists()


def test_api_files_duplicates_and_no_new_rows(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    df, _ = live.load_series(cfg, tmp_path)
    tail = [(d.date(), float(c)) for d, c in zip(df.index, df["close"], strict=True)][-2:]
    _api_file(tmp_path, tail)  # overlap only: nothing new
    full, sources, _, api_sha = live.load_series_full(cfg, tmp_path)
    assert len(full) == len(df) and "api" not in sources and api_sha is not None
    (csv,) = (tmp_path / live.API_DIR).glob("*.csv")
    csv.write_bytes(csv.read_bytes().rstrip(b"\n"))  # no trailing newline is accepted
    (tmp_path / live.API_MANIFEST).write_bytes(eodhd._manifest(tmp_path, RETRIEVED).encode())
    assert len(live.load_series(cfg, tmp_path)[0]) == len(df)
    _api_file(tmp_path, [tail[-1], (date(1996, 7, 1), 1500.0)])  # a second file, same date
    with pytest.raises(dl.DataValidationError, match="ascending and unique"):
        live.load_series(cfg, tmp_path)
