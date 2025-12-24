@echo off
title 🛡️ Risk Ops Console - Auto Launcher
setlocal

echo ==========================================
echo    INICIANDO RISK OPS CONSOLE
echo ==========================================
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"

REM ==========================================
REM 1. VERIFICAR/CREAR ENTORNO VIRTUAL
REM ==========================================

if not exist "venv" (
    echo [!] No se encontro el entorno virtual. Creandolo...
    echo.
    
    REM Intentar con py -3.11 primero (Windows Launcher)
    py -3.11 -m venv venv 2>nul
    
    if errorlevel 1 (
        REM Si falla, intentar con python directamente
        echo [!] Intentando con 'python' en lugar de 'py -3.11'...
        python -m venv venv
        
        if errorlevel 1 (
            echo [X] ERROR: No se pudo crear el entorno virtual
            echo.
            echo SOLUCION: Asegurate de tener Python 3.11 instalado
            echo Descargalo desde: https://www.python.org/downloads/
            echo.
            pause
            exit /b 1
        )
    )
    
    echo [OK] Entorno virtual creado exitosamente
    echo.
) else (
    echo [OK] Entorno virtual encontrado
)

REM ==========================================
REM 2. ACTIVAR ENTORNO VIRTUAL
REM ==========================================

echo [+] Activando entorno virtual...

if not exist "venv\Scripts\activate.bat" (
    echo [X] ERROR: El entorno virtual esta corrupto
    echo.
    echo SOLUCION: Elimina la carpeta 'venv' y vuelve a ejecutar este script
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if errorlevel 1 (
    echo [X] ERROR: No se pudo activar el entorno virtual
    pause
    exit /b 1
)

echo [OK] Entorno virtual activado
echo.

REM ==========================================
REM 3. VERIFICAR PYTHON
REM ==========================================

echo [+] Verificando version de Python...
python --version

if errorlevel 1 (
    echo [X] ERROR: Python no esta disponible en el entorno virtual
    pause
    exit /b 1
)

echo.

REM ==========================================
REM 4. INSTALAR/ACTUALIZAR DEPENDENCIAS
REM ==========================================

echo [+] Verificando dependencias...
echo    (esto puede tardar un momento la primera vez)
echo.

pip install -r requirements.txt --quiet --disable-pip-version-check

if errorlevel 1 (
    echo [!] WARNING: Hubo problemas al instalar algunas dependencias
    echo    El sistema intentara continuar de todas formas...
    echo.
)

echo [OK] Dependencias verificadas
echo.

REM ==========================================
REM 5. VERIFICAR ARCHIVOS CRITICOS
REM ==========================================

echo [+] Verificando archivos del sistema...

if not exist "console.py" (
    echo [X] ERROR: No se encontro console.py
    echo    Asegurate de estar en el directorio correcto del proyecto
    pause
    exit /b 1
)

if not exist "models" (
    echo [!] WARNING: Carpeta 'models' no encontrada, creandola...
    mkdir models
)

if not exist "models\risk_ops_backup.pkl" (
    echo [!] WARNING: Falta models\risk_ops_backup.pkl
    echo    El sistema intentara continuar con valores por defecto
)

if not exist "models\risk_ops_nn.keras" (
    echo [!] WARNING: Falta models\risk_ops_nn.keras
    echo    El sistema intentara continuar con valores por defecto
)

echo [OK] Verificacion completada
echo.

REM ==========================================
REM 6. LANZAR LA APLICACION
REM ==========================================

echo ==========================================
echo    EL SISTEMA ESTA LISTO
echo ==========================================
echo.
echo [*] Interfaz Web: http://127.0.0.1:7860
echo [*] Presiona CTRL+C para detener el servidor
echo [*] Cerrar esta ventana apagara el sistema
echo.
echo ==========================================
echo.

REM EJECUTAR CONSOLE.PY (no dashboard.py)
python console.py

REM ==========================================
REM 7. MANEJO DE ERRORES AL CERRAR
REM ==========================================

if errorlevel 1 (
    echo.
    echo ==========================================
    echo [X] ERROR: La aplicacion termino con errores
    echo ==========================================
    echo.
    echo POSIBLES CAUSAS:
    echo  * Faltan archivos de modelos en /models
    echo  * Puerto 7860 ya esta en uso
    echo  * Falta instalar TensorFlow
    echo.
    echo SOLUCION RAPIDA:
    echo  1. Revisa el log de errores arriba
    echo  2. Ejecuta: pip install tensorflow gradio
    echo  3. Verifica que los archivos .pkl y .keras existan en /models
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [OK] Sistema cerrado correctamente
echo ==========================================
echo.
pause
