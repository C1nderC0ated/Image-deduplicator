@echo off
rem  Explorer quotes a dropped path only when it contains a space, so cmd
rem  splits "D:\Photos&Videos" at the "&" (and "D:\Paris,2019" at the
rem  comma) before this file starts: %1 is D:\Photos, a DIFFERENT folder,
rem  which then got scanned - and "Videos" runs as a command after it. The
rem  untouched command line is kept for the Python side, which takes the
rem  real path from it (IMGDEDUP_RAWCMD), and a run started from Explorer
rem  ends with "exit", so nothing after the split can run. It is read with
rem  delayed expansion on, which is switched off again at once: on, it
rem  eats every "!" in a path.
setlocal EnableExtensions EnableDelayedExpansion
set "IMGDEDUP_RAWCMD=!cmdcmdline!"
set "IMGDEDUP_DROP="
rem  Substitutions work on the COPY: one on cmdcmdline itself rewrites
rem  cmdcmdline in place, and the next comparison reads the altered text.
set "_IMG_A=!IMGDEDUP_RAWCMD:%~nx0=!"
set "_IMG_B=!IMGDEDUP_RAWCMD: /c =!"
if not "!_IMG_A!"=="!IMGDEDUP_RAWCMD!" if not "!_IMG_B!"=="!IMGDEDUP_RAWCMD!" set "IMGDEDUP_DROP=1"
setlocal EnableExtensions DisableDelayedExpansion
rem  "call" expands its line twice, so a toolkit folder with "%" or "^" in
rem  its name broke every helper call. The helpers are called through this
rem  variable, written %%TOOLDIR%% so the second pass inserts it untouched.
set "TOOLDIR=%~dp0"
rem  A relative path means the folder it was typed in: made absolute here,
rem  before the cd below moves away from it.
set "TARGET="
if not "%~1"=="" set "TARGET=%~f1"
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem  Collect-Image-Inventory.bat
rem  Scans the folder this .bat sits in, or drag a folder onto it.
rem  READ-ONLY: writes image-inventory.jsonl and touches nothing else.
rem  Interpreter choice is delegated to _pick-python.bat so that every
rem  launcher in this folder agrees on which Python to use.
rem ----------------------------------------------------------------------

set "SCRIPT=%~dp0collect-image-inventory.py"
set "PROBE=from PIL import Image, ImageOps; Image.new('RGB',(2,2)).convert('L')"
rem a trailing backslash (e.g. a dragged drive root "D:\") would escape the
rem closing quote when passed on; "D:\." names the same folder and is safe.
rem The test sits below its own "if not defined" line. On the same line as
rem an "if defined" it ran with no argument too: cmd expands and parses a
rem whole line before running any of it, and the substring of an empty
rem TARGET is text it cannot parse, so every double-click start died here
rem with "The syntax of the command is incorrect." before printing a word.
if not defined TARGET goto :target_ready
if "%TARGET:~-1%"=="\" set "TARGET=%TARGET%."
:target_ready

if not exist "%SCRIPT%" (
    echo [FAIL] collect-image-inventory.py was not found next to this .bat.
    echo        Keep the whole folder together.
    echo.
    pause
    set "EXIT_CODE=1"
    goto :quit
)
if not exist "%~dp0_pick-python.bat" (
    echo [FAIL] _pick-python.bat is missing from this folder.
    echo.
    pause
    set "EXIT_CODE=1"
    goto :quit
)

call "%%TOOLDIR%%_pick-python.bat" "%PROBE%"
if not defined PYTHON_CMD goto :no_python
goto :run

:no_python
rem  No interpreter can run this stage. The shared handler offers
rem  setup, which detects the GPU and asks before installing.
call "%%TOOLDIR%%_offer-setup.bat" "%PROBE%"
if defined PYTHON_CMD goto :run
pause
set "EXIT_CODE=1"
goto :quit

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

:quit
rem  Every way out comes through here. A run started from Explorer ends
rem  with "exit", not "exit /b": a drop split at "&" leaves the rest of
rem  the name on cmd's line, and cmd runs it as a command once this file
rem  returns. The failure exits above used to return, so a "no Python" run
rem  on a folder named "Tom&Jerry" went on to run "Jerry".
if defined IMGDEDUP_DROP exit %EXIT_CODE%
exit /b %EXIT_CODE%
