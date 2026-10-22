@echo off
rem Builds viture_bridge.exe with the Visual Studio C++ tools (x64).
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" for /f "delims=" %%v in ('where /r "C:\Program Files\Microsoft Visual Studio" vcvars64.bat 2^>nul') do set "VCVARS=%%v"
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /EHsc /std:c++17 /W3 viture_bridge.cpp /Fe:viture_bridge.exe /link /SUBSYSTEM:CONSOLE
