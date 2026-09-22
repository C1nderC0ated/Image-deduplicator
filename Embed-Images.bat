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
rem  Embed-Images.bat
rem  Drag an image-inventory.jsonl (or its folder) onto this .bat.
rem  Computes CLIP embeddings. First run downloads the model (~600 MB).
rem
rem  Interpreter choice is delegated to _pick-python.bat - and it is done
rem  in TWO passes: first it looks for a Python whose torch can actually
rem  use a GPU, and only if none exists does it settle for a CPU torch. A
rem  CPU-only wheel cannot drive a GPU no matter what the code asks for,
rem  so preferring the right interpreter is the real fix.
rem ----------------------------------------------------------------------

set "SCRIPT=%~dp0embed-images.py"
set "PROBE=from PIL import Image; import torch, transformers; assert torch.__file__ and transformers.__file__; torch.zeros(1)"
rem  cuda covers AMD too - a ROCm build reuses the torch.cuda namespace and
rem  reports is_available() there. xpu is Intel Arc, and asking through
rem  getattr keeps this working on a torch too old to have the module.
rem  Apple Metal is not tested for: this file only runs on Windows. Testing
rem  cuda alone sent an Arc owner down the CPU path with a working GPU and
rem  a message saying so, which is the one thing the two passes exist to
rem  prevent.
set "PROBE_GPU=from PIL import Image; import torch, transformers; assert torch.__file__ and transformers.__file__; assert torch.cuda.is_available() or bool(getattr(torch, 'xpu', None)) and torch.xpu.is_available()"
rem With no argument this stage embeds the folder it sits in. That default
rem is set FIRST, so TARGET is never empty on the line below it. The other
rem way round, an "if defined" had to guard that line and could not: cmd
rem expands and parses a whole line before running any of it, and the
rem substring of an empty TARGET is text it cannot parse. Every start
rem without an argument, a double-click included, died here with
rem "The syntax of the command is incorrect." before printing anything.
rem A trailing backslash (e.g. a dragged drive root "D:\") would escape the
rem closing quote when passed on; "D:\." names the same folder and is safe.
if not defined TARGET set "TARGET=%~dp0."
if "%TARGET:~-1%"=="\" set "TARGET=%TARGET%."

if not exist "%SCRIPT%" (
    echo [FAIL] embed-images.py was not found next to this .bat.
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

rem -- pass 1: a Python whose torch can use the GPU right now ------------
call "%%TOOLDIR%%_pick-python.bat" "%PROBE_GPU%"
if not defined PYTHON_CMD goto :try_cpu
rem  which accelerator is left to the embedder, which names the actual
rem  device on its first line - the probe knows only that there is one
echo Using %PYTHON_CMD%  (GPU-capable torch)
echo.
goto :run

:try_cpu
call "%%TOOLDIR%%_pick-python.bat" "%PROBE%"
if not defined PYTHON_CMD goto :no_python
echo NOTE: no Python with a GPU-capable torch was found - embedding will
echo       run on the CPU. The embedder prints the exact reason and, if an
echo       Nvidia GPU is present, the reinstall command that fixes it.
echo.
echo Using %PYTHON_CMD%
echo.
goto :run

:no_python
call "%%TOOLDIR%%_offer-setup.bat" "%PROBE%"
if defined PYTHON_CMD goto :run
pause
set "EXIT_CODE=1"
goto :quit

:run
%PYTHON_CMD% "%SCRIPT%" "%TARGET%"
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
