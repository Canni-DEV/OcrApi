@echo off
REM Crea una tarea programada en Windows para ejecutar la API OCR al iniciar el sistema.
if "%~2"=="" (
    echo Uso: create_task.bat "C:\\Ruta\\a\\python.exe" "C:\\Ruta\\al\\proyecto"
    exit /b 1
)

set PYTHON_PATH=%~1
set PROJECT_PATH=%~2
set TASK_NAME=OCR_API_Service
set START_SCRIPT=%PROJECT_PATH%\scripts\start_api.bat

schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "\"%START_SCRIPT%\" \"%PYTHON_PATH%\"" ^
    /sc onstart ^
    /ru "SYSTEM" ^
    /RL HIGHEST ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo Tarea %TASK_NAME% creada correctamente.
) else (
    echo Error al crear la tarea programada.
)
