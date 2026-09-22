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
rem  Delayed expansion stays OFF from here on: with it on, "D:\Wow! Photos"
rem  arrived as "D:\Wow Photos" and "!NAME!" became a variable's value.
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem  Find-Duplicates.bat  -  the whole thing, once, on one folder.
rem
rem  Drag a folder onto this file (or double-click it to use the folder it
rem  sits in) and it runs all three stages in order, then opens the report.
rem  The per-stage .bat files still exist and still work; this is for the
rem  common case, which is "I have a folder of pictures, find the copies"
rem  and should not require knowing there are three stages at all.
rem
rem  The embedding stage is OPTIONAL and this says so rather than failing:
rem  without PyTorch the scan still finds duplicates by pixels, it just
rem  loses the semantic tier that catches recoloured and heavily cropped
rem  copies. So a machine with only Pillow gets a working tool, and is told
rem  what it is missing instead of being stopped by it.
rem
rem  Nothing is ever deleted here. The last stage writes a list and a
rem  separate recycler, and that recycler asks before it touches anything.
rem ----------------------------------------------------------------------

if not defined TARGET set "TARGET=%~dp0."
rem a trailing backslash (a dragged drive root, "D:\") would escape the
rem closing quote when passed on; "D:\." names the same folder and is safe
if "%TARGET:~-1%"=="\" set "TARGET=%TARGET%."

set "PROBE=from PIL import Image, ImageOps; Image.new('RGB',(2,2)).convert('L')"
rem  Functional, not name-only: _pick-python.bat requires it. A gutted
rem  install (folders survive, files deleted) still IMPORTS fine as an
rem  empty namespace package, and that has fooled this toolkit before.
rem  Same probe Embed-Images.bat and imgdedup.sh already use.
set "TPROBE=from PIL import Image; import torch, transformers; assert torch.__file__ and transformers.__file__; torch.zeros(1)"

for %%F in (collect-image-inventory.py analyze-inventory.py _pick-python.bat) do (
    if not exist "%~dp0%%F" (
        echo [FAIL] %%F was not found next to this .bat.
        echo        Keep the whole folder together.
        echo.
        pause
        set "EXIT_CODE=1"
        goto :quit
    )
)

call "%%TOOLDIR%%_pick-python.bat" "%PROBE%"
if not defined PYTHON_CMD (
    rem  Nothing here can even open an image. The shared handler detects the
    rem  GPU, shows the exact pip command and waits for a yes.
    call "%%TOOLDIR%%_offer-setup.bat" "%PROBE%"
)
if not defined PYTHON_CMD (
    pause
    set "EXIT_CODE=1"
    goto :quit
)
echo Using %PYTHON_CMD%
echo.

echo ======================================================================
echo   1 of 3   Scanning the folder
echo ======================================================================
rem  A native crash ends with a NEGATIVE exit code (0xC0000005 is
rem  -1073741819), and "if errorlevel 1" only means ">= 1": a crashed
rem  scan went on to embed and analyze half an inventory, exit 0.
rem  So every failure test here has a second line for below zero.
%PYTHON_CMD% "%~dp0collect-image-inventory.py" "%TARGET%"
rem  3 is collect's "no images in this folder": nothing to compare
if errorlevel 3 if not errorlevel 4 goto :no_images
if errorlevel 1 goto :failed
if not errorlevel 0 goto :failed

echo.
echo ======================================================================
echo   2 of 3   Reading the pictures with CLIP  (optional)
echo ======================================================================
set "TORCH_CMD="
call "%%TOOLDIR%%_pick-python.bat" "%TPROBE%"
rem  No outer quotes: PYTHON_CMD may itself be a quoted path, and
rem  set "X=%PYTHON_CMD%" put that path OUTSIDE the quoting - an "&" in it
rem  ended the command and the variable came out empty.
if defined PYTHON_CMD set TORCH_CMD=%PYTHON_CMD%
rem  _pick-python may have cleared PYTHON_CMD while probing for torch, so
rem  restore the interpreter that actually passed the Pillow probe before
rem  going on - stage 3 needs it whether or not stage 2 could run.
call "%%TOOLDIR%%_pick-python.bat" "%PROBE%"

if not defined TORCH_CMD goto :no_torch
%TORCH_CMD% "%~dp0embed-images.py" "%TARGET%"
if errorlevel 1 goto :embed_failed
if not errorlevel 0 goto :embed_failed
goto :stage3

:embed_failed
echo.
echo [skip] Embedding did not finish. Continuing with pixels only.
goto :stage3

:no_torch
echo PyTorch not installed for this Python - stage skipped.
echo Pixel comparison still runs. Without CLIP, recoloured and heavily
echo cropped copies are not detected. Run Check-Image-Tools.bat to add it.

:stage3

echo.
echo ======================================================================
echo   3 of 3   Comparing and writing the report
echo ======================================================================
rem  The FOLDER, not a file name: a second scan of a folder is written to
rem  image-inventory-2.jsonl, and naming image-inventory.jsonl here made
rem  every re-run embed and analyze the first scan - stale results, over
rem  the user's edited list, and in a copied folder a recycler aimed at
rem  the original. Given a folder, both stages take the newest inventory.
rem  The report's name moves with it, so analyze opens the one it wrote.
set "IMGDEDUP_OPEN_REPORT=1"
%PYTHON_CMD% "%~dp0analyze-inventory.py" "%TARGET%"
if errorlevel 1 goto :failed
if not errorlevel 0 goto :failed
set "IMGDEDUP_OPEN_REPORT="

echo.
pause
set "EXIT_CODE=0"
goto :quit

:no_images
echo.
echo No images in this folder - nothing to compare.
echo.
pause
set "EXIT_CODE=0"
goto :quit

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Stage failed. The later stages were not run.
echo.
pause

:quit
rem  Every way out comes through here, for the reason the header gives: a
rem  run started from Explorer ends with "exit", so nothing after a split
rem  can run. The two failure exits above the stages used to return with
rem  "exit /b", and after a split drop cmd then ran the rest of the name.
if defined IMGDEDUP_DROP exit %EXIT_CODE%
exit /b %EXIT_CODE%
