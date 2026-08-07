@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem  Embed-Images.bat
rem  Drag an image-inventory.jsonl (or its folder) onto this .bat.
rem  Computes CLIP embeddings. First run downloads the model (~600 MB).
rem
rem  Interpreter choice is delegated to _pick-python.bat - and it is done
rem  in TWO passes: first it looks for a Python whose torch can actually
rem  use a CUDA GPU, and only if none exists does it settle for a CPU
rem  torch. A CPU-only wheel cannot drive a GPU no matter what the code
rem  asks for, so preferring the right interpreter is the real fix.
rem ----------------------------------------------------------------------

set "SCRIPT=%~dp0embed-images.py"
set "PROBE=from PIL import Image; import torch, transformers; assert torch.__file__ and transformers.__file__; torch.zeros(1)"
set "PROBE_GPU=from PIL import Image; import torch, transformers; assert torch.__file__ and transformers.__file__ and torch.cuda.is_available()"
set "TARGET=%~1"
rem a trailing backslash (e.g. a dragged drive root "D:\") would escape the
rem closing quote when passed on; "D:\." names the same folder and is safe
if defined TARGET if "%TARGET:~-1%"=="\" set "TARGET=%TARGET%."
if not defined TARGET set "TARGET=%~dp0."

if not exist "%SCRIPT%" (
    echo [FAIL] embed-images.py was not found next to this .bat.
    echo        Keep the whole folder together.
    echo.
    pause
    exit /b 1
)
if not exist "%~dp0_pick-python.bat" (
    echo [FAIL] _pick-python.bat is missing from this folder.
    echo.
    pause
    exit /b 1
)

rem -- pass 1: a Python whose torch can use the GPU right now ------------
call "%~dp0_pick-python.bat" "%PROBE_GPU%"
if not defined PYTHON_CMD goto :try_cpu
echo Using %PYTHON_CMD%  (CUDA-capable torch)
echo.
goto :run

:try_cpu
call "%~dp0_pick-python.bat" "%PROBE%"
if not defined PYTHON_CMD goto :no_python
echo NOTE: no Python with a CUDA-capable torch was found - embedding will
echo       run on the CPU. The embedder prints the exact reason and, if an
echo       Nvidia GPU is present, the reinstall command that fixes it.
echo.
echo Using %PYTHON_CMD%
echo.
goto :run

:no_python
call "%~dp0_offer-setup.bat" "%PROBE%"
if defined PYTHON_CMD goto :run
pause
exit /b 1

:run
%PYTHON_CMD% "%SCRIPT%" "%TARGET%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
pause
exit /b %EXIT_CODE%
