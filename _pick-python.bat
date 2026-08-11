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
rem     3. .venv beside this file - what setup builds when it has to make
rem        one. Uncommon on Windows (that is mainly the PEP 668 route, and
rem        Windows ships no marker) but reachable: setup offers the venv
rem        whenever pip is MISSING and venv works, which is not gated on
rem        platform. Without this step setup could install every package
rem        into a .venv the launchers then refused to look at, and every
rem        later run reported those same packages missing while the fix sat
rem        on disk.
rem     4. python                 whatever is on PATH
rem
rem  Deliberately BELOW the py launcher, unlike imgdedup.sh where .venv
rem  comes first. On Linux a PEP 668 distro forces everything into the venv,
rem  so it has to win; on Windows `py` is the idiomatic entry point and a
rem  stray .venv should not quietly hijack it. It still sits above bare
rem  `python`, which is the weakest signal here - it can be a Store stub or
rem  whatever happens to be first on PATH.
rem
rem  The ordering only decides between candidates that BOTH pass the probe,
rem  so this costs nothing in the case that motivated the step: when setup
rem  installed into .venv because system pip was broken, the py-launcher
rem  interpreters fail the functional probe and fall through to it anyway.
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
if errorlevel 1 goto :pp_venv
for %%V in (3.14 3.13 3.12 3.11 3.10 3.9) do call :pp_try_ver %%V
if defined PYTHON_CMD goto :pp_done

rem  That list has an expiry date, so ask `py` for its own default before
rem  giving up on the launcher. When 3.15 ships, `py -3.15` is not tried by
rem  the loop above and this file would fall through to .venv and then bare
rem  `python` - which on Windows is often a Store stub or missing entirely,
rem  so a machine whose only Python was 3.15 could be told there is none
rem  while `py -3.15` sat there working. `py -3` means "newest Python 3
rem  this launcher knows about" and needs no editing ever.
rem
rem  It stays BELOW the explicit list rather than replacing it: the list is
rem  ordered newest-first on purpose and each entry is probed functionally,
rem  so if the newest interpreter cannot import what a stage needs, an older
rem  one still wins. `py -3` alone would take the newest and stop.
py -3 -c "%_PP_PROBE%" >nul 2>&1
if errorlevel 1 goto :pp_venv
set "PYTHON_CMD=py -3"
goto :pp_done

rem %~dp0 is THIS file's folder - the toolkit folder - and already ends in a
rem backslash. It is deliberately not the caller's directory: the .venv sits
rem beside the scripts, not beside whatever was dragged onto them.
:pp_venv
if not exist "%~dp0.venv\Scripts\python.exe" goto :pp_path
"%~dp0.venv\Scripts\python.exe" -c "%_PP_PROBE%" >nul 2>&1
if errorlevel 1 goto :pp_path
set "PYTHON_CMD="%~dp0.venv\Scripts\python.exe""
goto :pp_done

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
