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
rem  which accelerator is left to the embedder, which names the actual
rem  device on its first line - the probe knows only that there is one
echo Using %PYTHON_CMD%  (GPU-capable torch)
echo.
goto :run

:try_cpu
call "%~dp0_pick-python.bat" "%PROBE%"
if not defined PYTHON_CMD goto :no_python
echo NOTE: no Python with a GPU-capable torch was found - embedding will
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
