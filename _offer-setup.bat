@echo off
rem ----------------------------------------------------------------------
rem  _offer-setup.bat  -  shared "you are missing something" handler
rem
rem  Every launcher calls THIS when _pick-python.bat found no interpreter
rem  that can actually run its stage. One place, so the three launchers
rem  cannot drift apart again - and the thing that CHANGES the machine is
rem  the last place to allow that.
rem
rem  It never installs anything itself: it finds a Python and hands off to
rem  _setup.py, which detects the GPU, asks which PyTorch build you want,
rem  and shows every command before running it.
rem
rem  Usage:  call "%%TOOLDIR%%_offer-setup.bat" "<probe code>"
rem  On return PYTHON_CMD is set if the stage can now run.
rem
rem  No setlocal: PYTHON_CMD must survive back into the caller.
rem ----------------------------------------------------------------------
set "_OS_PROBE=%~1"
set "_OS_PY="
if not defined TOOLDIR set "TOOLDIR=%~dp0"

rem  A Python counts only if it RUNS. "where python" also finds the
rem  Microsoft Store stub that stock Windows ships with no Python at all,
rem  and a new user was told "A Python is installed" and handed a command
rem  that could not work. Exactly 0 passes: a crash exits below zero.
rem  The override is held to the same test. Taken on trust, one naming a
rem  file that does not exist gave the same false "A Python is installed"
rem  and a setup command that could not start.
if not defined IMGDEDUP_PYTHON goto :os_launcher
"%IMGDEDUP_PYTHON:"=%" -c "import sys" >nul 2>&1
if errorlevel 0 if not errorlevel 1 set _OS_PY="%IMGDEDUP_PYTHON:"=%"
:os_launcher
if not defined _OS_PY (
    py -3 -c "import sys" >nul 2>&1
    if errorlevel 0 if not errorlevel 1 set "_OS_PY=py -3"
)
if not defined _OS_PY (
    python -c "import sys" >nul 2>&1
    if errorlevel 0 if not errorlevel 1 set "_OS_PY=python"
)

if not defined _OS_PY (
    rem No Python at all - nothing to install INTO. Show what each said.
    call "%%TOOLDIR%%_why-no-python.bat" "%_OS_PROBE%"
    exit /b 1
)

if not exist "%~dp0_setup.py" (
    echo.
    echo [FAIL] _setup.py is missing from this folder - keep the toolkit
    echo        together, or run Check-Image-Tools.bat for manual commands.
    echo.
    exit /b 1
)

echo.
echo ======================================================================
echo  A Python is installed, but this stage is missing some packages.
echo.
echo  Setup can install them. It detects your GPU (NVIDIA / AMD / Intel),
echo  asks which PyTorch build you want, and shows every command before
echo  running it - nothing is installed without your say-so.
echo ======================================================================
echo.
set "_OS_ANS="
set /p _OS_ANS="Run setup now? (Y/N): "
rem  Quotes come out before the answer is compared: a lone double quote
rem  typed here unbalanced the comparison, and the whole launcher died with
rem  "The syntax of the command is incorrect." The strip has its own guard
rem  line: on an empty answer it would put a stray quote IN instead.
if not defined _OS_ANS goto :os_skip
set "_OS_ANS=%_OS_ANS:"=%"
if /I "%_OS_ANS%"=="Y" goto :os_run
if /I "%_OS_ANS%"=="YES" goto :os_run
:os_skip
echo.
echo Skipped. You can run it any time:
echo    %_OS_PY% "%~dp0_setup.py"
echo.
exit /b 1

:os_run
%_OS_PY% "%~dp0_setup.py"
echo.
rem Re-probe: the stage may be runnable now.
call "%%TOOLDIR%%_pick-python.bat" "%_OS_PROBE%"
if defined PYTHON_CMD (
    echo Setup finished - continuing.
    echo.
    exit /b 0
)
echo.
echo Still not runnable. Run Check-Image-Tools.bat for a full report.
echo.
exit /b 1
