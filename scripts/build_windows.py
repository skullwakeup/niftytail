"""Build the Windows app (D69): dist\\NIFTYTAIL-windows\\, its zip and dist\\NIFTYTAIL-setup.exe.

Run: ``uv run --locked python scripts/build_windows.py --uv-zip <uv-x86_64-pc-windows-msvc.zip>``

The zip must be the official uv 0.12.15 release (sha256 of D68). The app is a `git archive` of
HEAD restricted to the include list (tracked files only, no tests, no publish tooling); the tree
must be clean so that HEAD is what ships. The installer (packaging/niftytail.iss) is compiled
only if Inno Setup 6's ISCC.exe is found.
"""

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "NIFTYTAIL-windows"
UV_ZIP_SHA256 = "477bd99a84e34891f2bd4c9152ddeb74e971accccbc59c0f0301f11f08a32d46"  # D68 (1)
UV_EXE_SHA256 = "b0131eb55f112aee1836951ba96abec9c897b7bedd3f2fdebb19d710cfb636cd"  # D68 (1)
LAUNCHER = "packaging/NIFTYTAIL.cmd"
ISS = ROOT / "packaging" / "niftytail.iss"
# D68 include list, plus pyproject.toml (PyCrucible used to add it; uv run needs it)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Windows app (D69).")
    parser.add_argument("--uv-zip", type=Path, required=True)
    args = parser.parse_args()

    if git("status", "--porcelain").strip():
        print("refused: the working tree is not clean")
        return 1
    uv_zip = args.uv_zip.read_bytes()
    if sha256(uv_zip) != UV_ZIP_SHA256:
        print(f"refused: {args.uv_zip} sha256 {sha256(uv_zip)} is not the D68 value")
        return 1
    uv_exe = zipfile.ZipFile(io.BytesIO(uv_zip)).read("uv.exe")
    if sha256(uv_exe) != UV_EXE_SHA256:
        print("refused: uv.exe in the zip is not the D68 binary")
        return 1

    tracked = git("ls-tree", "-r", "--name-only", "-z", "HEAD").decode().split("\0")
    files = app_files([p for p in tracked if p])
    # git archive applies .gitattributes, so NIFTYTAIL.cmd gets its CRLF line endings
    tar = git("archive", "--format=tar", "HEAD", "--", LAUNCHER, *files)
    if OUT.exists():
        shutil.rmtree(OUT)
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        archive.extractall(OUT / "app", filter="data")
    (OUT / "app" / LAUNCHER).replace(OUT / "NIFTYTAIL.cmd")
    (OUT / "uv.exe").write_bytes(uv_exe)
    zip_path = shutil.make_archive(str(OUT), "zip", root_dir=DIST, base_dir=OUT.name)
    print(f"{len(files)} app files; wrote {OUT} and {zip_path}")

    iscc = find_iscc()
    if iscc is None:
        print("ISCC.exe (Inno Setup 6) not found: the installer step was skipped")
        return 0
    subprocess.run([iscc, "/Q", str(ISS)], check=True)
    print(f"wrote {DIST / 'NIFTYTAIL-setup.exe'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
