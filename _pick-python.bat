@echo off
rem ----------------------------------------------------------------------
rem  _pick-python.bat   -   shared interpreter chooser
rem
rem  Every launcher in this folder calls THIS, so they all agree on which
rem  Python to use. Change the order here and it changes everywhere.
rem
rem  Usage:   call "%~dp0_pick-python.bat" "<probe code>"
rem  Sets:    PYTHON_CMD   (already quoted when it is a path)
rem           PICK_ERR     1 when nothing matched, 0 otherwise
rem
rem  Order, most trusted first:
rem     1. %IMGDEDUP_PYTHON%      explicit override, always wins
rem     2. py -3.14 .. -3.9       the launcher, newest first
rem     3. python                 whatever is on PATH
rem
rem  Every probe passed in here must be FUNCTIONAL - it has to call into
rem  the package, not just name it: a gutted install (folders survive,
rem  files deleted) still imports fine as an empty namespace package, and
rem  that fooled this toolkit once.
rem
rem  No setlocal: this runs in the caller's variable scope on purpose, so
rem  PYTHON_CMD survives the return without endlocal tricks.
rem ----------------------------------------------------------------------
set "_PP_PROBE=%~1"
set "PYTHON_CMD="
set "PICK_ERR=1"

if not defined IMGDEDUP_PYTHON goto :pp_launcher
"%IMGDEDUP_PYTHON%" -c "%_PP_PROBE%" >nul 2>&1
if errorlevel 1 goto :pp_launcher
set "PYTHON_CMD="%IMGDEDUP_PYTHON%""
goto :pp_done

:pp_launcher
where py >nul 2>&1
if errorlevel 1 goto :pp_path
for %%V in (3.14 3.13 3.12 3.11 3.10 3.9) do call :pp_try_ver %%V
if defined PYTHON_CMD goto :pp_done

:pp_path
python -c "%_PP_PROBE%" >nul 2>&1
if errorlevel 1 goto :pp_fail
set "PYTHON_CMD=python"
goto :pp_done

:pp_try_ver
if defined PYTHON_CMD exit /b 0
py -%1 -c "%_PP_PROBE%" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHON_CMD=py -%1"
exit /b 0

:pp_fail
set "PYTHON_CMD="
set "PICK_ERR=1"
exit /b 1

:pp_done
set "PICK_ERR=0"
exit /b 0
