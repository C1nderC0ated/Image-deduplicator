@echo off
rem  Command line kept for the Python side, "!" safe, "%" and "^" safe in
rem  helper calls, relative paths resolved: see Find-Duplicates.bat.
setlocal EnableExtensions EnableDelayedExpansion
set "IMGDEDUP_RAWCMD=!cmdcmdline!"
set "IMGDEDUP_DROP="
rem  Substitutions work on the COPY: one on cmdcmdline itself rewrites
rem  cmdcmdline in place, and the next comparison reads the altered text.
set "_IMG_A=!IMGDEDUP_RAWCMD:%~nx0=!"
set "_IMG_B=!IMGDEDUP_RAWCMD: /c =!"
if not "!_IMG_A!"=="!IMGDEDUP_RAWCMD!" if not "!_IMG_B!"=="!IMGDEDUP_RAWCMD!" set "IMGDEDUP_DROP=1"
setlocal EnableExtensions DisableDelayedExpansion
set "TOOLDIR=%~dp0"
set "TARGET="
if not "%~1"=="" set "TARGET=%~f1"
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
    echo [FAIL] analyze-inventory.py was not found next to this .bat.
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
