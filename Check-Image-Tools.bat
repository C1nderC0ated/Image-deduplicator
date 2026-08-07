@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem  Check-Image-Tools.bat
rem  Read-only doctor: reports every Python on this machine and what each
rem  can actually do, then prints the exact command to fix what is missing.
rem  Interpreter choice is delegated to _pick-python.bat so that every
rem  launcher in this folder agrees on which Python to use.
rem ----------------------------------------------------------------------

set "SCRIPT=%~dp0check-image-tools.py"
set "PROBE=import sys"

if not exist "%SCRIPT%" (
    echo [FAIL] check-image-tools.py was not found next to this .bat.
    echo        Keep the whole folder together.
    echo.
    pause
    exit /b 1
)
if not exist "%~dp0_pick-python.bat" (
    echo [FAIL] _pick-python.bat is missing from this folder.
    echo.
    pause
    exit /b 1
)

call "%~dp0_pick-python.bat" "%PROBE%"
if not defined PYTHON_CMD goto :no_python
goto :run

:no_python
call "%~dp0_why-no-python.bat" "%PROBE%"
pause
exit /b 1

:run
echo Using %PYTHON_CMD%
echo.
%PYTHON_CMD% "%SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
pause
exit /b %EXIT_CODE%
