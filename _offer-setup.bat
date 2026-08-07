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
rem  Usage:  call "%~dp0_offer-setup.bat" "<probe code>"
rem  On return PYTHON_CMD is set if the stage can now run.
rem
rem  No setlocal: PYTHON_CMD must survive back into the caller.
rem ----------------------------------------------------------------------
set "_OS_PROBE=%~1"
set "_OS_PY="

if defined IMGDEDUP_PYTHON set "_OS_PY="%IMGDEDUP_PYTHON%""
if not defined _OS_PY (
    where py >nul 2>&1
    if not errorlevel 1 set "_OS_PY=py -3"
)
if not defined _OS_PY (
    where python >nul 2>&1
    if not errorlevel 1 set "_OS_PY=python"
)

if not defined _OS_PY (
    rem No Python at all - nothing to install INTO. Show what each said.
    call "%~dp0_why-no-python.bat" "%_OS_PROBE%"
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
if /I "%_OS_ANS%"=="Y" goto :os_run
if /I "%_OS_ANS%"=="YES" goto :os_run
echo.
echo Skipped. You can run it any time:
echo    %_OS_PY% "%~dp0_setup.py"
echo.
exit /b 1

:os_run
%_OS_PY% "%~dp0_setup.py"
echo.
rem Re-probe: the stage may be runnable now.
call "%~dp0_pick-python.bat" "%_OS_PROBE%"
if defined PYTHON_CMD (
    echo Setup finished - continuing.
    echo.
    exit /b 0
)
echo.
echo Still not runnable. Run Check-Image-Tools.bat for a full report.
echo.
exit /b 1
