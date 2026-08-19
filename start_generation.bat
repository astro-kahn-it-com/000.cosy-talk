@echo off
setlocal

:: Lock environment variables locally
set "TEMP=%~dp0cache\temp"
set "TMP=%~dp0cache\temp"
set "PIP_CACHE_DIR=%~dp0cache\pip"
set "HF_HOME=%~dp0models"
set "MODELS_DIR=%~dp0models"

:: Create local containment folders
if not exist "%~dp0cache\temp" mkdir "%~dp0cache\temp"
if not exist "%~dp0cache\pip" mkdir "%~dp0cache\pip"
if not exist "%~dp0models" mkdir "%~dp0models"
if not exist "%~dp0output" mkdir "%~dp0output"
if not exist "%~dp0voices" mkdir "%~dp0voices"

:: Run generate.py and pause on exit for logging
"%~dp0ComfyUI_windows_portable\python_embeded\python.exe" "%~dp0generate.py"
pause
