import re
from pathlib import Path

from agent_graph.app import UI_DIR


def test_the_static_ui_has_no_build_step_no_cdn_no_innerhtml_and_no_secrets():
    files = [p for p in Path(UI_DIR).iterdir() if p.is_file()]
    assert {p.name for p in files} >= {"index.html", "app.js"}
    for path in files:
        text = path.read_text()
        assert not re.search(r"https?://(?!www\.w3\.org)", text), f"{path.name} loads from the web"
        assert not re.search(r"\.innerHTML\s*=|insertAdjacentHTML", text), path.name
        assert not re.search(r"sk-[A-Za-z0-9]{10,}|APP_API_KEY=", text), path.name
