@echo off
rem Builds myviture.exe (native Windows app) with the Visual Studio C++ tools (x64).
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" for /f "delims=" %%v in ('where /r "C:\Program Files\Microsoft Visual Studio" vcvars64.bat 2^>nul') do set "VCVARS=%%v"
call "%VCVARS%" >nul 2>nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /EHsc /std:c++20 /W3 /utf-8 myviture.cpp /Fe:myviture.exe /link /SUBSYSTEM:CONSOLE
