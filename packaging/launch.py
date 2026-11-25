"""Windows app entrypoint (D67, D69, D74): the dashboard in public demo mode.

Run by packaging/NIFTYTAIL.cmd with the bundled Python (`python.exe -I`, D74). Sets
NIFTYTAIL_PUBLIC=1, changes to the project root (the dashboard reads tracked files and
.streamlit/config.toml relative to it) and starts Streamlit headless on port 8501 (no first-run
e-mail question), bound to localhost: no external-IP lookup at start and not reachable from the
network (D74). A background thread opens the browser once the health endpoint answers "ok",
polling for up to 120 s.
"""

import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from streamlit.web import cli

URL = "http://localhost:8501"
WAIT_S = 120


def _open_when_ready() -> None:
    # no proxy: the health check goes to this machine only
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        try:
            with opener.open(URL + "/_stcore/health", timeout=2) as response:
                if response.read().strip() == b"ok":
                    webbrowser.open(URL)
                    return
        except OSError:
            pass
        time.sleep(0.5)


def main() -> None:
    os.environ["NIFTYTAIL_PUBLIC"] = "1"
    os.chdir(Path(__file__).resolve().parents[1])
    threading.Thread(target=_open_when_ready, daemon=True).start()
    sys.argv = [
        "streamlit",
        "run",
        "app/dashboard.py",
        "--server.headless=true",
        "--server.port=8501",
        "--server.address=localhost",
    ]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
