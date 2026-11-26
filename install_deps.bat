@echo off
echo ============================================
echo LeafTrack-SG Dependency Installer
echo ============================================
echo.

set PIP="%~dp0venv\Scripts\python.exe"

echo [1/4] Installing image + data packages...
%PIP% -m pip install opencv-python scikit-learn matplotlib seaborn tqdm pandas
if %errorlevel% neq 0 goto :error

echo [2/4] Installing report + utility packages...
%PIP% -m pip install reportlab requests pytest gdown
if %errorlevel% neq 0 goto :error

echo [3/4] Installing TensorFlow (this may take several minutes)...
%PIP% -m pip install tensorflow==2.17.0
if %errorlevel% neq 0 (
    echo TensorFlow 2.17.0 failed. Trying tensorflow-cpu...
    %PIP% -m pip install tensorflow-cpu
    if %errorlevel% neq 0 (
        echo WARNING: TensorFlow could not be installed.
        echo The website will work but AI inference will fail.
        echo Try Python 3.12 for full TensorFlow support.
    ) else (
        echo tensorflow-cpu installed successfully.
    )
) else (
    echo TensorFlow 2.17.0 installed successfully.
)

echo [4/4] Verifying key imports...
%PIP% -c "import flask; print('Flask OK:', flask.__version__)"
%PIP% -c "import numpy; print('NumPy OK:', numpy.__version__)"
%PIP% -c "import PIL; print('Pillow OK:', PIL.__version__)"
%PIP% -c "import cv2; print('OpenCV OK:', cv2.__version__)"
%PIP% -c "import sklearn; print('scikit-learn OK:', sklearn.__version__)"
%PIP% -c "import reportlab; print('ReportLab OK:', reportlab.Version)"

echo.
echo ============================================
echo Installation complete!
echo.
echo To activate the venv, run:
echo   venv\Scripts\activate.bat
echo.
echo Then start the website:
echo   python app.py
echo ============================================
goto :end

:error
echo ERROR: Installation failed at step above.
pause
exit /b 1

:end
echo Done.
pause
