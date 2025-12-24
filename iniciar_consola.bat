@echo off
title 🛡️ Risk Ops Console - Auto Launcher
setlocal

echo ==========================================
echo    INICIANDO RISK OPS CONSOLE
echo ==========================================

:: 1. Verificar si existe la carpeta venv
if not exist "venv" (
    echo [!] No se encontro el entorno virtual. Creandolo...
    py -3.11 -m venv venv
)

:: 2. Activar el entorno virtual
echo [+] Activando entorno virtual...
call .\venv\Scripts\activate

:: 3. Instalar/Actualizar requerimientos
echo [+] Verificando dependencias (esto puede tardar un momento)...
pip install -r requirements.txt --quiet

:: 4. Lanzar el Dashboard
echo.
echo ==========================================
echo    EL SISTEMA ESTA LISTO
echo    Cerrar esta ventana apagara el Dashboard
echo ==========================================
echo.
python dashboard.py

pause