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
set "_WN_OV=%IMGDEDUP_PYTHON:"=%"
if not defined _WN_OV goto :wn_override_shim
if /i "%_WN_OV:~-4%"==".bat" goto :wn_override_shim
if /i "%_WN_OV:~-4%"==".cmd" goto :wn_override_shim
"%IMGDEDUP_PYTHON:"=%" -c "%PROBE%" 2>&1
echo.
goto :wn_launcher
:wn_override_shim
echo It is empty or names a batch file. Neither can be used as the Python
echo here: set IMGDEDUP_PYTHON to python.exe itself.
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
if errorlevel 1 goto :wn_no_path
set "_WN_EXT="
set "_WN_SEEN="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined _WN_SEEN (
    set "_WN_SEEN=1"
    set "_WN_EXT=%%~xP"
)
echo --- python on PATH ---
if /i "%_WN_EXT%"==".bat" goto :wn_path_shim
if /i "%_WN_EXT%"==".cmd" goto :wn_path_shim
python -c "%PROBE%" 2>&1
echo.
goto :wn_end
:wn_path_shim
echo It is a batch file, which cannot be used as the Python here. pyenv-win
echo puts python.bat on PATH; set IMGDEDUP_PYTHON to the real python.exe,
echo which "pyenv which python" prints.
echo.
goto :wn_end
:wn_no_path
echo --- no python on PATH ---
echo.
:wn_end
echo ======================================================================
echo  Run Check-Image-Tools.bat - it lists every Python on this machine
echo  and prints the exact pip command to fix whichever you want to use.
echo ======================================================================
echo.
endlocal
exit /b 0
