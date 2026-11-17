@echo off
rem Day 13F (D60 (f)), order per D61 (a): the end-of-day update. Works from cmd, PowerShell and the Task Scheduler:
rem   scripts\update.cmd [run dir]
rem validate-live (only if data\live exists), fetch-api (EODHD), forecast,
rem forecast --check-run <run dir>. On success each step prints only "<step>: exit 0"; on a
rem non-zero exit it prints that command's own output (no dates or key by design) and stops.
setlocal
cd /d "%~dp0.."
set "RUN=%~1"
if "%RUN%"=="" set "RUN=reports/runs/e65780dc4f81b1cad5a3bdd88b0fd18bb69f9ba3585655814af338174c0b28c9"
set "LOG=%TEMP%\niftytail-update-%RANDOM%%RANDOM%.log"
if not exist "data\live\" goto fetch
call :step validate-live validate-live
if errorlevel 1 exit /b %errorlevel%
:fetch
call :step fetch-api fetch-api
if errorlevel 1 exit /b %errorlevel%
call :step forecast forecast
if errorlevel 1 exit /b %errorlevel%
call :step check-run forecast --check-run "%RUN%"
exit /b %errorlevel%

:step
set "NAME=%~1"
uv run --locked python -m niftytail %2 configs/base.toml %3 %4 >"%LOG%" 2>&1
set "RC=%errorlevel%"
echo %NAME%: exit %RC%
if not "%RC%"=="0" type "%LOG%"
del "%LOG%" >nul 2>&1
exit /b %RC%
