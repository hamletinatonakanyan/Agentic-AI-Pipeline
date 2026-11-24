"""Live test for the MCP server: `make test-live`.

It needs Node (`npx`) and downloads the official MCP Inspector the first time, so it is slow and
needs the network. It checks that an independent MCP client can read our stdio server. If the
Inspector's command-line flags change, adjust the arguments below. For an interactive session,
run `make inspect-mcp`.
"""

import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.live


def test_the_official_inspector_can_list_our_tools_over_stdio():
    npx = shutil.which("npx")
    if not npx:
        pytest.skip("npx (Node) is not installed")
    done = subprocess.run(
        [npx, "-y", "@modelcontextprotocol/inspector", "--cli", sys.executable,
         "-m", "mcp_tools.server", "--method", "tools/list"],
        capture_output=True, text=True, timeout=180,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-500:]
    assert "calculate" in done.stdout and "search_corpus" in done.stdout
