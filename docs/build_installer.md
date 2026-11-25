# Building the Windows installer (D74)

How to build `NIFTYTAIL-setup.exe`, the offline Windows app of README "Windows app". The installer contains everything the app needs: a Python, the packages and the dashboard files. Nothing is downloaded when the app is installed or run.

## Requirements

- Windows 10 or 11, 64-bit, with Git.
- uv 0.12.15 (the version that wrote `uv.lock`) and Python 3.13 installed on the PC: the build script runs with `uv run`, which uses the installed Python (`python-preference = "only-system"` in `pyproject.toml`).
- Inno Setup 6 (`winget install --id JRSoftware.InnoSetup -e`; 6.7.3 was used). The build looks for `ISCC.exe` on PATH and in the "Inno Setup 6" folders only. If winget now offers Inno Setup 7, install a 6.x release instead.
- Internet at build time: the build downloads the Python package from python.org and the wheels from PyPI (uv takes the wheels from its cache when they are already there).
- A clean working tree: the app is built from the committed HEAD, so uncommitted changes would not ship. The build refuses otherwise.

## Build

From the repository root:

```powershell
uv run --locked python scripts/build_windows.py
```

If python.org cannot be reached from the build PC, download https://www.python.org/ftp/python/3.13.16/python-3.13.16-embed-amd64.zip elsewhere and pass it:

```powershell
uv run --locked python scripts/build_windows.py --python-zip <path to python-3.13.16-embed-amd64.zip>
```

Either way the zip must have sha256 `97dae5274cc54867065e8d5a3226e48c35017ed332a0fdb0e27d5b5821961297` (the value python.org publishes for this file); any other file is refused.

## Outputs

All under `dist\` (gitignored; binaries are never committed):

| file | what it is | measured on the D74 build |
|---|---|---|
| `NIFTYTAIL-windows\` | the app folder; the installed app is this folder | 12,327 files, 520,727,885 bytes (before the first run) |
| `NIFTYTAIL-windows.zip` | the same folder as a zip, to run without installing | 172,814,908 bytes |
| `NIFTYTAIL-setup.exe` | the installer (per-user, no administrator rights) | 109,851,066 bytes |

The build ends by printing the installer's sha256 and the build time (80 s on the development PC for the D74 build, with the wheels already in the uv cache).

The installer and its uninstaller are not code-signed. Where Smart App Control is on, Windows may block them, online and offline (seen on the development PC, D74). On such PCs use the zip: extract it and double-click `NIFTYTAIL.cmd`, which starts the bundled `python.exe`, signed by the Python Software Foundation. Code signing (for example through the SignPath Foundation, free for open-source projects) is not done.

## What the build does

1. Refuses unless the working tree is clean.
2. Gets the CPython 3.13.16 embeddable package (64-bit) from python.org, or from `--python-zip`, and checks its sha256. This is the official package, signed by the Python Software Foundation; uv-managed interpreters are blocked by Windows Application Control on the development PC (D24).
3. Writes `NIFTYTAIL-windows\app\`: a `git archive` of HEAD restricted to the files the public dashboard reads (the include list in `scripts/build_windows.py`; no tests, no publish tooling). `NIFTYTAIL.cmd` moves to the top of the folder.
4. Unpacks the Python package into `NIFTYTAIL-windows\python\` and replaces `python313._pth` with the bundle's import path: `python313.zip`, `.`, `Lib\site-packages`, `..\app\src`, then `import site`. With this file present, Python ignores PYTHONPATH and PYTHONHOME, but on its own it still enables the user's site-packages (`sys.flags.no_user_site` is 0, checked on the development PC); the launcher's `-I` turns that off.
5. Installs the packages: `uv export --locked --no-dev --group dashboard --no-emit-project` writes the locked dashboard set with its hashes, and `uv pip install --target python\Lib\site-packages --python-platform x86_64-pc-windows-msvc --python-version 3.13.16 --only-binary :all: --require-hashes --no-deps` installs exactly those Windows wheels. Nothing is resolved anew and nothing is built from source. The niftytail package itself is not installed: Python imports it from `app\src`.
6. Smoke test (Windows only): the bundled `python.exe -I`, run from `app\` like the launcher, imports streamlit, pyarrow, pandas, numpy, scipy, statsmodels, matplotlib, altair, niftytail.dashboard_data and niftytail.demo, checks that it runs isolated, that niftytail comes from `app\src` and that no import path lies outside the bundle, and prints `smoke test: imports ok`. It runs before the zip and the installer are made, so the `__pycache__` folders it writes ship with them (the first start compiles less).
7. Writes `NIFTYTAIL-windows.zip`.
8. Reads the version from `pyproject.toml` and compiles `packaging/niftytail.iss` with `ISCC /Q /DAppVersion=<version>` into `NIFTYTAIL-setup.exe` (Windows only, if Inno Setup 6 is found; otherwise it prints that the installer step was skipped). The script refuses to compile without a version.

On Linux or macOS the build runs up to the zip (the wheels are Windows wheels whatever the build machine); the smoke test and the installer are skipped with a printed line.

The launcher `NIFTYTAIL.cmd` changes to `app\` and runs `python\python.exe -I packaging\launch.py`. `-I` (isolated mode) also switches off the user's site-packages, which the `._pth` file alone leaves on. `launch.py` starts Streamlit headless on localhost:8501 (no external-IP lookup at start, not reachable from other machines) and opens the browser when the app answers.

## Checking that it works offline

1. Turn Wi-Fi off and unplug the network cable.
2. Run `NIFTYTAIL-setup.exe` with "Launch NIFTYTAIL" ticked: a console window opens, then the browser at http://localhost:8501. Open all four sections.
3. Close the console, start the app again from the Start menu.
4. Uninstall (Settings, Apps), then check that `Test-Path "$env:LOCALAPPDATA\Programs\NIFTYTAIL"` gives False.

A PC that has never had Python or NIFTYTAIL installed is the strongest test.

## Changing versions

- **A package version:** change `pyproject.toml`, run `uv lock`, and check that the dashboard still loads on Windows with Smart App Control on: some newer compiled wheels are blocked there (D24, D56). `app/requirements.txt` pins streamlit and pyarrow for Streamlit Cloud to the `uv.lock` versions (a test checks it). Then rebuild; the installer takes the new wheels from the lock.
- **The Python version:** in `scripts/build_windows.py`, set `PYTHON_VERSION` and `PYTHON_ZIP_SHA256` to the new embeddable package (64-bit) and its sha256 from the python.org release page. For a new minor version (3.14), the `._pth` file and `python313.zip` change name, and `requires-python`, `.python-version` and the wheels must all agree. 3.13.16 is the last 3.13 release with Windows binaries.
- **The app version:** `version` in `pyproject.toml` is the one source. Change it and `__version__` in `src/niftytail/__init__.py` (a test checks that they agree), run `uv lock` (only niftytail's own line changes), commit, and rebuild: the installer shows the new version. Tag the published commit with the same number (`v1.2.0` for 1.2.0).
- **Inno Setup:** `scripts/build_windows.py` looks for Inno Setup 6 only. A move to another major version needs `find_iscc` and a test compile of `packaging/niftytail.iss`.

## Troubleshooting

| message or symptom | cause | what to do |
|---|---|---|
| `refused: the working tree is not clean` | uncommitted changes | commit or stash them, then build again |
| `refused: the Python zip has sha256 ...` | not the pinned file (another version, 32-bit, or a damaged download) | download the exact file above again, or use `--python-zip` with a checked copy |
| download error at the start | python.org not reachable | download the zip elsewhere and use `--python-zip` |
| `uv export` fails with a lock error | `uv.lock` does not match `pyproject.toml` | run `uv lock`, review the change, commit |
| `uv pip install` cannot find a wheel | a locked package has no Windows cp313 wheel, or PyPI is not reachable | build with internet, or change that package's version |
| the smoke test fails on an import | a DLL blocked by Smart App Control, or a missing file | compare with the project's own `uv` environment, which uses the same wheels (same hashes); see D24 and D56 |
| `ISCC.exe (Inno Setup 6, Windows) not found` | Inno Setup 6 not installed, or only version 7 | install Inno Setup 6 |
| ISCC: `AppVersion is not defined` | `niftytail.iss` compiled by hand | build with `scripts/build_windows.py`, or pass `/DAppVersion=<version>` |
| the app window shows an error and waits | the app stopped (for example port 8501 in use) | close the other program on port 8501 or the earlier NIFTYTAIL window, then start again |
