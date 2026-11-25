"""Build the offline Windows app (D69, D74): dist\\NIFTYTAIL-windows\\, its zip and the installer.

Run: ``uv run --locked python scripts/build_windows.py [--python-zip <embeddable zip>]``

The folder holds NIFTYTAIL.cmd, python\\ (the official CPython 3.13.16 embeddable package, sha256
pinned below, with every locked dashboard package installed from its hash-checked Windows wheel)
and app\\ (a `git archive` of HEAD restricted to the include list: tracked files only, no tests,
no publish tooling). The tree must be clean so that HEAD is what ships. The Python zip is
downloaded from python.org unless --python-zip gives a copy; any other sha256 is refused. Steps
that need Windows (the smoke test of the bundled Python, ISCC) are skipped elsewhere with a
printed line; the installer (packaging/niftytail.iss) is compiled only if Inno Setup 6's
ISCC.exe is found, with the version of pyproject.toml.
"""

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "NIFTYTAIL-windows"
SETUP = DIST / "NIFTYTAIL-setup.exe"
# D74 (1): the official embeddable package; the hash is the one python.org publishes
PYTHON_VERSION = "3.13.16"
PYTHON_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
)
PYTHON_ZIP_SHA256 = "97dae5274cc54867065e8d5a3226e48c35017ed332a0fdb0e27d5b5821961297"
# D74 (3): the import path of the bundled Python, nothing from the environment
PTH = "python313.zip\r\n.\r\nLib\\site-packages\r\n..\\app\\src\r\n\r\nimport site\r\n"
LAUNCHER = "packaging/NIFTYTAIL.cmd"
ISS = ROOT / "packaging" / "niftytail.iss"
# D68 include list, plus pyproject.toml (the version source; D69 pin (1))
INCLUDE = (
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "src/niftytail/*.py",
    "src/niftytail/py.typed",
    "packaging/launch.py",
    "app/dashboard.py",
    "app/demo/*.json",
    "app/static/fonts/*",
    ".streamlit/config.toml",
    "configs/base.toml",
    "reports/oos/**/*.json",
    "reports/figures/*.svg",
    "docs/results_review.md",
    "docs/figures.md",
)
EXCLUDE = ("src/niftytail/publish.py",)
# D74 (6): run by the bundled python.exe -I from app\, as the launcher does
SMOKE = """
import importlib, sys
from pathlib import Path
for name in ("streamlit", "pyarrow", "pandas", "numpy", "scipy", "statsmodels", "matplotlib",
             "altair", "niftytail.dashboard_data", "niftytail.demo"):
    importlib.import_module(name)
import niftytail
root = Path(sys.executable).resolve().parents[1]
print("sys.flags.isolated", sys.flags.isolated, "sys.flags.no_user_site", sys.flags.no_user_site)
print("sys.path", sys.path)
assert sys.flags.isolated == 1 and sys.flags.no_user_site == 1
assert Path(niftytail.__file__).resolve().is_relative_to(root / "app" / "src"), niftytail.__file__
outside = [p for p in sys.path if not Path(p).resolve().is_relative_to(root)]
assert not outside, outside
print("smoke test: imports ok")
"""


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def app_files(tracked: list[str]) -> list[str]:
    def hit(path: str, patterns: tuple[str, ...]) -> bool:
        return any(PurePosixPath(path).full_match(p) for p in patterns)

    return sorted(p for p in tracked if hit(p, INCLUDE) and not hit(p, EXCLUDE))


def find_iscc() -> str | None:
    # PATH, then the per-user (winget) and the machine-wide install folders of Inno Setup 6
    homes = [Path(os.environ.get("LOCALAPPDATA", ""), "Programs")]
    homes += [Path(os.environ[v]) for v in ("ProgramFiles(x86)", "ProgramFiles") if v in os.environ]
    found = [p for p in (h / "Inno Setup 6" / "ISCC.exe" for h in homes) if p.is_file()]
    return shutil.which("ISCC") or (str(found[0]) if found else None)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def folder_size(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def install_packages(site_packages: Path) -> None:
    # D74 (2): the locked dashboard set as Windows cp313 wheels, hash-checked, no resolution
    uv = os.environ.get("UV", "uv")
    export = (
        "export --locked --no-dev --group dashboard --no-emit-project --format requirements.txt"
    )
    pip = (
        f"pip install --python-platform x86_64-pc-windows-msvc --python-version {PYTHON_VERSION}"
        " --only-binary :all: --require-hashes --no-deps"
    )
    with tempfile.TemporaryDirectory() as tmp:
        req = str(Path(tmp, "requirements.txt"))
        # --quiet: with -o, uv export would also echo the whole file to stdout
        subprocess.run([uv, *export.split(), "--quiet", "-o", req], cwd=ROOT, check=True)
        target = ["--target", str(site_packages), "-r", req]
        subprocess.run([uv, *pip.split(), *target], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the offline Windows app (D74).")
    parser.add_argument("--python-zip", type=Path, help="a local copy of the embeddable zip")
    args = parser.parse_args()
    start = time.monotonic()
    sys.stdout.reconfigure(line_buffering=True)  # keeps the order with the subprocess output

    if git("status", "--porcelain").strip():
        print("refused: the working tree is not clean")
        return 1
    if args.python_zip is None:
        print(f"downloading {PYTHON_URL}")
        with urllib.request.urlopen(PYTHON_URL, timeout=60) as response:
            python_zip = response.read()
    else:
        python_zip = args.python_zip.read_bytes()
    if sha256(python_zip) != PYTHON_ZIP_SHA256:
        print(f"refused: the Python zip has sha256 {sha256(python_zip)}, not the D74 value")
        return 1
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]

    tracked = git("ls-tree", "-r", "--name-only", "-z", "HEAD").decode().split("\0")
    files = app_files([p for p in tracked if p])
    # git archive applies .gitattributes, so NIFTYTAIL.cmd gets its CRLF line endings
    tar = git("archive", "--format=tar", "HEAD", "--", LAUNCHER, *files)
    if OUT.exists():
        shutil.rmtree(OUT)
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        archive.extractall(OUT / "app", filter="data")
    (OUT / "app" / LAUNCHER).replace(OUT / "NIFTYTAIL.cmd")
    python = OUT / "python"
    zipfile.ZipFile(io.BytesIO(python_zip)).extractall(python)
    (python / "python313._pth").write_bytes(PTH.encode("ascii"))
    install_packages(python / "Lib" / "site-packages")
    print(f"niftytail {version}: {len(files)} app files, Python {PYTHON_VERSION}")

    # D74 pin: the smoke test runs before zipping, so its __pycache__ folders ship
    if sys.platform == "win32":
        exe = str(python / "python.exe")
        subprocess.run([exe, "-I", "-c", SMOKE], cwd=OUT / "app", check=True)
    else:
        print("not Windows: the smoke test of the bundled Python was skipped")
    count, size = folder_size(OUT)
    print(f"wrote {OUT}: {count} files, {size} bytes")
    zip_path = Path(shutil.make_archive(str(OUT), "zip", root_dir=DIST, base_dir=OUT.name))
    print(f"wrote {zip_path}: {zip_path.stat().st_size} bytes")

    iscc = find_iscc() if sys.platform == "win32" else None
    if iscc is None:
        print("ISCC.exe (Inno Setup 6, Windows) not found: the installer step was skipped")
    else:
        subprocess.run([iscc, "/Q", f"/DAppVersion={version}", str(ISS)], check=True)
        data = SETUP.read_bytes()
        print(f"wrote {SETUP}: {len(data)} bytes, sha256 {sha256(data)}")
    print(f"build time {time.monotonic() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
