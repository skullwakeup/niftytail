@echo off
rem NIFTYTAIL Windows app (D74): the dashboard in public demo mode, offline. The bundled Python
rem (python\) runs isolated (-I) and imports only what is in this folder.
setlocal
cd /d "%~dp0app"
"%~dp0python\python.exe" -I packaging\launch.py
if not "%ERRORLEVEL%"=="0" pause
