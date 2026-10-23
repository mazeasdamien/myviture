@echo off
rem Builds myviture.exe (native Windows app) with the Visual Studio C++ tools (x64).
rem Needs the downloaded parts first (ONNX Runtime, DirectML, hand models): fetch_deps.bat.
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" for /f "delims=" %%v in ('where /r "C:\Program Files\Microsoft Visual Studio" vcvars64.bat 2^>nul') do set "VCVARS=%%v"
call "%VCVARS%" >nul 2>nul || exit /b 1
cd /d "%~dp0"
if not exist third_party\onnxruntime\lib\onnxruntime.lib (echo Missing ONNX Runtime: run fetch_deps.bat & exit /b 1)
if not exist obj mkdir obj
set IMGUI=third_party\imgui
rem Dear ImGui (MIT, v1.92.9b) changes rarely: compiled once into obj\.
if not exist obj\imgui.obj cl /nologo /c /O2 /EHsc /std:c++20 /utf-8 /I%IMGUI% /Foobj\ %IMGUI%\imgui.cpp %IMGUI%\imgui_draw.cpp %IMGUI%\imgui_tables.cpp %IMGUI%\imgui_widgets.cpp %IMGUI%\imgui_demo.cpp %IMGUI%\backends\imgui_impl_win32.cpp %IMGUI%\backends\imgui_impl_dx11.cpp || exit /b 1
rem /Zi + /DEBUG: myviture.pdb, so a crash's stack (crash.txt) names the functions and lines.
cl /nologo /O2 /Zi /EHsc /std:c++20 /W3 /utf-8 /I%IMGUI% /Foobj\ /Fdobj\ myviture.cpp dashboard.cpp obj\imgui*.obj /Fe:myviture.exe /link /SUBSYSTEM:CONSOLE /DEBUG /OPT:REF /OPT:ICF || exit /b 1
rem Runtime DLLs next to the exe (DirectML.dll must be this one, not Windows' older copy).
copy /y third_party\onnxruntime\lib\onnxruntime.dll . >nul
copy /y third_party\onnxruntime\lib\onnxruntime_providers_shared.dll . >nul
copy /y third_party\directml\DirectML.dll . >nul
