@echo off
rem ----------------------------------------------------------------------
rem  _why-no-python.bat  -  shared failure report
rem  Called when _pick-python.bat finds nothing. Prints what each candidate
rem  ACTUALLY said, instead of a useless "not found".
rem  Usage:  call "%~dp0_why-no-python.bat" "<probe code>"
rem ----------------------------------------------------------------------
setlocal EnableExtensions DisableDelayedExpansion
set "PROBE=%~1"

echo.
echo ======================================================================
echo  No suitable Python was found. Here is what each candidate said:
echo ======================================================================
echo.
rem  Quotes stripped, as _pick-python.bat does it. A value set WITH quotes
rem  around a path holding a parenthesis, such as "C:\Program Files (x86)",
rem  switched the quoting off around its own path. The ")" then closed the
rem  block this line used to sit in, and the report died at once with an
rem  "unexpected at this time" error. Out of the block and under its own
rem  guard line, it also never meets an empty variable.
if not defined IMGDEDUP_PYTHON goto :wn_launcher
echo --- IMGDEDUP_PYTHON override ---
"%IMGDEDUP_PYTHON:"=%" -c "%PROBE%" 2>&1
echo.
:wn_launcher
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo --- py -3 ---
    py -3 -c "%PROBE%" 2>&1
    echo.
) else (
    echo --- py launcher not installed ---
    echo.
)
if exist "%~dp0.venv\Scripts\python.exe" (
    echo --- .venv beside the toolkit ---
    "%~dp0.venv\Scripts\python.exe" -c "%PROBE%" 2>&1
    echo.
) else (
    echo --- no .venv beside the toolkit ---
    echo.
)
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo --- python on PATH ---
    python -c "%PROBE%" 2>&1
    echo.
) else (
    echo --- no python on PATH ---
    echo.
)
echo ======================================================================
echo  Run Check-Image-Tools.bat - it lists every Python on this machine
echo  and prints the exact pip command to fix whichever you want to use.
echo ======================================================================
echo.
endlocal
exit /b 0
