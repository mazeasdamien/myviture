@echo off
rem Downloads what the native app needs but git does not keep (binaries and models):
rem   ONNX Runtime DirectML 1.24.4 (MIT, nuget.org), DirectML 1.15.4 (Microsoft, nuget.org),
rem   MediaPipe palm and hand landmark models, ONNX from opencv_zoo (Apache 2.0).
setlocal
cd /d "%~dp0"
set TMPD=%TEMP%\myviture-deps
if exist "%TMPD%" rmdir /s /q "%TMPD%"
mkdir "%TMPD%"
curl -sL -o "%TMPD%\ort.zip" https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.24.4/microsoft.ml.onnxruntime.directml.1.24.4.nupkg || exit /b 1
curl -sL -o "%TMPD%\dml.zip" https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg || exit /b 1
powershell -NoProfile -Command "Expand-Archive -Force '%TMPD%\ort.zip' '%TMPD%\ort'; Expand-Archive -Force '%TMPD%\dml.zip' '%TMPD%\dml'" || exit /b 1
mkdir third_party\onnxruntime\include third_party\onnxruntime\lib third_party\directml models 2>nul
copy /y "%TMPD%\ort\build\native\include\*.h" third_party\onnxruntime\include >nul
copy /y "%TMPD%\ort\runtimes\win-x64\native\*" third_party\onnxruntime\lib >nul
copy /y "%TMPD%\ort\LICENSE" third_party\onnxruntime\LICENSE >nul
copy /y "%TMPD%\dml\bin\x64-win\DirectML.dll" third_party\directml >nul
copy /y "%TMPD%\dml\LICENSE.txt" third_party\directml >nul
set ZOO=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
curl -sL -o models\palm_detection_mediapipe_2023feb.onnx %ZOO%/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx || exit /b 1
curl -sL -o models\handpose_estimation_mediapipe_2023feb.onnx %ZOO%/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx || exit /b 1
rmdir /s /q "%TMPD%"
echo done
