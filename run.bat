@echo off
rem Launch the mcp-soundfx server.
rem Set MCP_SOUNDFX_VENV to your virtualenv path, or create .venv next to this script.
if "%MCP_SOUNDFX_VENV%"=="" (set "VENV_DIR=%~dp0.venv") else (set "VENV_DIR=%MCP_SOUNDFX_VENV%")
call "%VENV_DIR%\Scripts\activate.bat"
python "%~dp0mcp_soundfx.py" %*
