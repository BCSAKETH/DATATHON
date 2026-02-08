@echo off
echo ========================================
echo                AgriMinds
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH!
    echo Please install Node.js and try again.
    pause
    exit /b 1
)

echo [1/3] Starting Backend Server (FastAPI)...
echo.
cd backend
start "AgriMinds Backend" cmd /k "python main.py"
timeout /t 3 /nobreak >nul

echo [2/3] Starting Frontend Server (Node.js)...
echo.
cd ..\frontend
start "AgriMinds Frontend" cmd /k "npm run dev"
timeout /t 5 /nobreak >nul

echo [3/3] Opening Application in Browser...
echo.
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo    AgriMinds is now running!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5172
echo.
echo Press any key to keep this window open...
echo To stop the servers, close the Backend and Frontend windows.
pause >nul
