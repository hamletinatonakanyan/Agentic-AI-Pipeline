from common import Environment


def test_the_real_environment_wins_over_the_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("MY_KEY=from-file\n")
    assert Environment(f, {"MY_KEY": "from-env"}).get("MY_KEY") == "from-env"


def test_the_env_file_is_used_when_the_variable_is_not_exported(tmp_path):
    f = tmp_path / ".env"
    f.write_text("MY_KEY=from-file\nOTHER=x\n")
    env = Environment(f, {})
    assert env.get("MY_KEY") == "from-file" and env.get("OTHER") == "x"
    assert env.get("MISSING") is None and env.get("MISSING", "fallback") == "fallback"


def test_empty_values_count_as_unset(tmp_path):
    f = tmp_path / ".env"
    f.write_text("EMPTY=\n")
    assert Environment(f, {"EMPTY": ""}).get("EMPTY") is None


def test_a_missing_env_file_is_fine(tmp_path):
    assert Environment(tmp_path / "nope.env", {}).get("X") is None


def test_the_file_is_read_each_time_so_edits_are_seen(tmp_path):
    f = tmp_path / ".env"
    env = Environment(f, {})
    assert env.get("K") is None
    f.write_text("K=now-set\n")
    assert env.get("K") == "now-set"


def test_values_with_quotes_and_comments_parse_like_dotenv(tmp_path):
    f = tmp_path / ".env"
    f.write_text('A="quoted value"\nB=plain  # a comment\n')
    env = Environment(f, {})
    assert env.get("A") == "quoted value" and env.get("B") == "plain"
