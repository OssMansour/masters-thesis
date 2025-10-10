@echo off
REM ========================================================================
REM Script to Move Workspace from OneDrive to Local Drive (C:\Projects)
REM This improves performance and reliability for machine learning training
REM ========================================================================

echo.
echo ========================================
echo   Workspace Migration Tool
echo   OneDrive -^> C:\Projects
echo ========================================
echo.

REM Create C:\Projects directory
echo [1/4] Creating C:\Projects directory...
if not exist "C:\Projects" (
    mkdir "C:\Projects"
    echo      ✓ Created C:\Projects
) else (
    echo      ✓ C:\Projects already exists
)

echo.
echo [2/4] Copying masters-thesis workspace...
echo      This may take 5-10 minutes depending on size...
echo.

REM Copy the entire workspace
xcopy /E /I /Y "C:\Users\LEGION\OneDrive\Old files\Documents\masters-thesis" "C:\Projects\masters-thesis"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo      ✓ Workspace copied successfully!
) else (
    echo.
    echo      ✗ Copy failed! Error code: %ERRORLEVEL%
    pause
    exit /b 1
)

echo.
echo [3/4] Verifying copy...
if exist "C:\Projects\masters-thesis\SAC_drone\SAC_gym_pybullet.py" (
    echo      ✓ Verification successful!
) else (
    echo      ✗ Verification failed - files not found!
    pause
    exit /b 1
)

echo.
echo [4/4] Setup complete!
echo.
echo ========================================
echo   Migration Complete!
echo ========================================
echo.
echo Your workspace is now at:
echo   C:\Projects\masters-thesis
echo.
echo Old location (OneDrive):
echo   C:\Users\LEGION\OneDrive\Old files\Documents\masters-thesis
echo.
echo NEXT STEPS:
echo   1. Close this window
echo   2. Open new terminal in: C:\Projects\masters-thesis\SAC_drone
echo   3. Activate environment: conda activate SAC
echo   4. Start training: python SAC_gym_pybullet.py
echo.
echo ========================================
echo.
pause
