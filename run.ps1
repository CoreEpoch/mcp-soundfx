# Launch the mcp-soundfx server.
# Set MCP_SOUNDFX_VENV to your virtualenv path, or create .venv next to this script.
$VenvDir = if ($env:MCP_SOUNDFX_VENV) { $env:MCP_SOUNDFX_VENV } else { Join-Path $PSScriptRoot '.venv' }
& "$VenvDir\Scripts\python.exe" "$PSScriptRoot\mcp_soundfx.py" $args
