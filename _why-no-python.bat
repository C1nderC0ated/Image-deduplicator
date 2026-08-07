@echo off
rem ----------------------------------------------------------------------
rem  _why-no-python.bat  -  shared failure report
rem  Called when _pick-python.bat finds nothing. Prints what each candidate
rem  ACTUALLY said, instead of a useless "not found".
rem  Usage:  call "%~dp0_why-no-python.bat" "<probe code>"
rem ----------------------------------------------------------------------
setlocal EnableExtensions
set "PROBE=%~1"

echo.
echo ======================================================================
echo  No suitable Python was found. Here is what each candidate said:
echo ======================================================================
echo.
if defined IMGDEDUP_PYTHON (
    echo --- IMGDEDUP_PYTHON override ---
    "%IMGDEDUP_PYTHON%" -c "%PROBE%" 2>&1
    echo.
)
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo --- py -3 ---
    py -3 -c "%PROBE%" 2>&1
    echo.
) else (
    echo --- py launcher not installed ---
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
