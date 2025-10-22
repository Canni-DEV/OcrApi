@echo off
REM Script para iniciar la API OCR usando Uvicorn
REM Uso: start_api.bat [ruta_completa_python]
set PYTHON_EXE=%~1
if "%PYTHON_EXE%"=="" (
    set PYTHON_EXE=python
)

cd /d %~dp0\..
"%PYTHON_EXE%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
