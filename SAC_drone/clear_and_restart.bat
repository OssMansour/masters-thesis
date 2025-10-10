@echo off
echo ================================================================================
echo Clearing Old SAC Training Data
echo ================================================================================
echo.
echo This will delete:
echo   - logs/SAC/Drone/best_model/
echo   - logs/best_training_model/
echo   - logs/final/
echo   - logs/tensorboard/SAC_fresh/
echo   - logs/sac_spiral/
echo   - logs/training_log.txt
echo   - logs/eval_log.txt
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Deleting old models and logs...

REM Delete model directories
if exist "logs\SAC\Drone\best_model" rmdir /s /q "logs\SAC\Drone\best_model"
if exist "logs\best_training_model" rmdir /s /q "logs\best_training_model"
if exist "logs\final" rmdir /s /q "logs\final"
if exist "logs\tensorboard\SAC_fresh" rmdir /s /q "logs\tensorboard\SAC_fresh"
if exist "logs\sac_spiral" rmdir /s /q "logs\sac_spiral"

REM Delete log files
if exist "logs\training_log.txt" del /q "logs\training_log.txt"
if exist "logs\eval_log.txt" del /q "logs\eval_log.txt"

REM Recreate directories
echo.
echo Recreating fresh directories...
mkdir "logs\SAC\Drone\best_model"
mkdir "logs\best_training_model"
mkdir "logs\final"
mkdir "logs\tensorboard\SAC_fresh"

echo.
echo ================================================================================
echo ✓ Cleanup complete! Ready for fresh training.
echo ================================================================================
echo.
echo To start training, run:
echo   python SAC_gym_pybullet.py
echo.
pause
