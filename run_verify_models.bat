@echo off
REM Runs the model verification with the project Python (scikit-learn 1.3.2, same as production)
cd /d "%~dp0"
echo Running FitVision model verification...
"C:\fit\.conda\python.exe" tools\testing\verify_models.py > data\evaluation\verify_models_log.txt 2>&1
echo exit code %ERRORLEVEL% >> data\evaluation\verify_models_log.txt
type data\evaluation\verify_models_log.txt
echo.
echo Done. You can close this window.
