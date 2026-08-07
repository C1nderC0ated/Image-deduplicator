@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem  Analyze-Inventory.bat
rem  Drag an image-inventory.jsonl (or its folder) onto this .bat.
rem  Finds duplicates and writes a report, a selection list and a
rem  Recycle-Bin script. READ-ONLY: deletes nothing itself.
rem  Interpreter choice is delegated to _pick-python.bat so that every
rem  launcher in this folder agrees on which Python to use.
rem ----------------------------------------------------------------------

set "SCRIPT=%~dp0analyze-inventory.py"
set "PROBE=import numpy, PIL; from PIL import Image; Image.new('RGB',(2,2))"
set "TARGET=%~1"
rem a trailing backslash (e.g. a dragged drive root "D:\") would escape the
rem closing quote when passed on; "D:\." names the same folder and is safe
if defined TARGET if "%TARGET:~-1%"=="\" set "TARGET=%TARGET%."

if not exist "%SCRIPT%" (
    echo [FAIL] analyze-inventory.py was not found next to this .bat.
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
call "%~dp0_offer-setup.bat" "%PROBE%"
if defined PYTHON_CMD goto :run
pause
exit /b 1

:run
echo Using %PYTHON_CMD%
echo.
if defined TARGET (
    %PYTHON_CMD% "%SCRIPT%" "%TARGET%"
) else (
    %PYTHON_CMD% "%SCRIPT%"
)
set "EXIT_CODE=%ERRORLEVEL%"

echo.
pause
exit /b %EXIT_CODE%
