@echo off
setlocal
if defined VIDEO_PYTHON (
  set "DEMO_PYTHON=%VIDEO_PYTHON%"
) else (
  set "DEMO_PYTHON=G:\Books\Python\ACTUAL CODES\PROJECTS\Assignments\recruit_ai_app\.venv\Scripts\python.exe"
)
if not exist "%DEMO_PYTHON%" (
  echo Playwright Python environment not found: %DEMO_PYTHON%
  echo Install requirements-video.txt into a Python environment and set VIDEO_PYTHON.
  exit /b 1
)
"%DEMO_PYTHON%" -m scripts.record_demo %*
