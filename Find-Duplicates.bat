@echo off
setlocal EnableExtensions EnableDelayedExpansion
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

set "TARGET=%~1"
if not defined TARGET set "TARGET=%~dp0."
rem a trailing backslash (a dragged drive root, "D:\") would escape the
rem closing quote when passed on; "D:\." names the same folder and is safe
if "%TARGET:~-1%"=="\" set "TARGET=%TARGET%."

set "PROBE=from PIL import Image, ImageOps; Image.new('RGB',(2,2)).convert('L')"
set "TPROBE=import torch, transformers"

for %%F in (collect-image-inventory.py analyze-inventory.py _pick-python.bat) do (
    if not exist "%~dp0%%F" (
        echo [FAIL] %%F was not found next to this .bat.
        echo        Keep the whole folder together.
        echo.
        pause
        exit /b 1
    )
)

call "%~dp0_pick-python.bat" "%PROBE%"
if not defined PYTHON_CMD (
    rem  Nothing here can even open an image. The shared handler detects the
    rem  GPU, shows the exact pip command and waits for a yes.
    call "%~dp0_offer-setup.bat" "%PROBE%"
)
if not defined PYTHON_CMD (
    pause
    exit /b 1
)
echo Using %PYTHON_CMD%
echo.

set "INV=%TARGET%\image-inventory.jsonl"

echo ======================================================================
echo   1 of 3   Scanning the folder
echo ======================================================================
%PYTHON_CMD% "%~dp0collect-image-inventory.py" "%TARGET%"
if errorlevel 1 goto :failed

echo.
echo ======================================================================
echo   2 of 3   Reading the pictures with CLIP  (optional)
echo ======================================================================
set "TORCH_CMD="
call "%~dp0_pick-python.bat" "%TPROBE%"
if defined PYTHON_CMD set "TORCH_CMD=%PYTHON_CMD%"
rem  _pick-python may have cleared PYTHON_CMD while probing for torch, so
rem  restore the interpreter that actually passed the Pillow probe before
rem  going on - stage 3 needs it whether or not stage 2 could run.
call "%~dp0_pick-python.bat" "%PROBE%"

if defined TORCH_CMD (
    %TORCH_CMD% "%~dp0embed-images.py" "%INV%"
    if errorlevel 1 (
        echo.
        echo [skip] The embedding stage did not finish. Carrying on without
        echo        it - the scan below still works, with pixels only.
    )
) else (
    echo PyTorch is not installed for this Python, so this stage is skipped.
    echo The scan still finds duplicates by pixel comparison. What it loses
    echo is the semantic pass, which is what catches a recoloured or heavily
    echo cropped copy. Run Check-Image-Tools.bat to add it.
)

echo.
echo ======================================================================
echo   3 of 3   Comparing and writing the report
echo ======================================================================
%PYTHON_CMD% "%~dp0analyze-inventory.py" "%INV%"
if errorlevel 1 goto :failed

echo.
for %%R in ("%TARGET%\duplicates-report.html") do if exist "%%~fR" (
    echo Opening the report. Mark what you want gone, download the list over
    echo the old one, then run the recycler beside it.
    start "" "%%~fR"
)
echo.
pause
exit /b 0

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo That stage failed, so the later ones were not run.
echo.
pause
exit /b %EXIT_CODE%
