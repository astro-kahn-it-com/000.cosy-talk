@echo off
setlocal

:: Lock all environment variables locally
set "ROOT_DIR=%~dp0"
set "TEMP=%ROOT_DIR%cache\temp"
set "TMP=%ROOT_DIR%cache\temp"
set "PIP_CACHE_DIR=%ROOT_DIR%cache\pip"
set "HF_HOME=%ROOT_DIR%cache\hf"
set "MODELS_DIR=%ROOT_DIR%models"

:: Create local containment folders
if not exist "%ROOT_DIR%cache\temp" mkdir "%ROOT_DIR%cache\temp"
if not exist "%ROOT_DIR%cache\pip" mkdir "%ROOT_DIR%cache\pip"
if not exist "%ROOT_DIR%cache\hf" mkdir "%ROOT_DIR%cache\hf"
if not exist "%ROOT_DIR%models" mkdir "%ROOT_DIR%models"
if not exist "%ROOT_DIR%output" mkdir "%ROOT_DIR%output"
if not exist "%ROOT_DIR%voices" mkdir "%ROOT_DIR%voices"

:: Point directly to ComfyUI python_embeded
set "PYTHON_EXE=%ROOT_DIR%ComfyUI_windows_portable\python_embeded\python.exe"

:: Run generate.py
"%PYTHON_EXE%" "%ROOT_DIR%generate.py"

:: Pause on exit for logging
pause
endlocal
