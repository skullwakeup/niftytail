@echo off
rem NIFTYTAIL Windows app (D69): the dashboard in public demo mode. Everything uv downloads
rem (packages, and Python 3.13 if none is installed) stays inside this folder.
setlocal
set "NIFTYTAIL_PUBLIC=1"
set "UV_CACHE_DIR=%~dp0cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0python"
cd /d "%~dp0app"
echo NIFTYTAIL: the first run downloads packages and takes a few minutes; later runs start quickly.
"%~dp0uv.exe" run --python-preference system --locked --no-dev --group dashboard python packaging\launch.py
if not "%ERRORLEVEL%"=="0" pause
