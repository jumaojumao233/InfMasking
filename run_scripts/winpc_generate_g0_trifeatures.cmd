@echo off
setlocal

set "PROJECT=%~dp0.."
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"
set "DATA_ROOT=%PROJECT%\dataset\data\trifeatures_g0_seed20260924"
set "STDOUT_LOG=%PROJECT%\g0_generation.stdout.log"
set "STDERR_LOG=%PROJECT%\g0_generation.stderr.log"
set "STATUS_LOG=%PROJECT%\g0_generation.status.log"

pushd "%PROJECT%"
echo START=%DATE% %TIME%> "%STATUS_LOG%"
"%PYTHON%" -u "%PROJECT%\run_scripts\generate_trifeatures_dataset.py" --root "%DATA_ROOT%" --seed 20260924 1> "%STDOUT_LOG%" 2> "%STDERR_LOG%"
set "EXIT_CODE=%ERRORLEVEL%"
echo END=%DATE% %TIME% EXIT_CODE=%EXIT_CODE%>> "%STATUS_LOG%"
popd
exit /b %EXIT_CODE%
