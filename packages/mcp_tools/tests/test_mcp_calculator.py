import pytest

from mcp_tools.calculator import MAX_EXPONENT, MAX_LENGTH, CalculatorError, calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 1", 2),
        ("2 * (3 + 4)", 14),
        ("2 + 3 * 4", 14),  # precedence
        ("(2 + 3) * 4", 20),
        ("10 / 4", 2.5),
        ("10 // 4", 2),
        ("10 % 4", 2),
        ("2 ** 10", 1024),
        ("2 ** 3 ** 2", 512),  # right associative
        ("-3 + 5", 2),
        ("+3", 3),
        ("- -3", 3),
        ("0.1 + 0.2", pytest.approx(0.3)),
        ("1e3 + 1", 1001),
        ("  7  ", 7),
        ("2 ** -1", 0.5),
        ("1_000 * 2", 2000),
    ],
)
def test_ordinary_arithmetic(expression, expected):
    assert calculate(expression) == expected


def test_the_result_is_always_a_float():
    assert isinstance(calculate("2 + 2"), float)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1 / 0", "divide by zero"),
        ("1 // 0", "divide by zero"),
        ("5 % 0", "divide by zero"),
        ("", "empty"),
        ("   ", "empty"),
        ("2 +", "not a valid arithmetic"),
        ("2 2", "not a valid arithmetic"),
        ("(1 + 2", "not a valid arithmetic"),
        ("2 ** 100000", "exponent is too large"),
        ("9 ** 9 ** 9", "exponent is too large"),
        ("10 ** 400", "too large"),
        ("1e308 * 10", "too large"),
        ("1e308 * 10 - 1e308 * 10", "too large"),  # inf - inf is nan
        ("(-8) ** 0.5", "not a real number"),
    ],
)
def test_clear_errors_for_things_that_go_wrong(expression, message):
    with pytest.raises(CalculatorError, match=message):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('x = 1')",
        "abs(-1)",  # even harmless calls are refused
        "pi",
        "x + 1",
        "(1).real",
        "'a' * 3",
        "'abc'",
        "[1, 2][0]",
        "{1: 2}",
        "lambda: 1",
        "[x for x in range(3)]",
        "1 if True else 2",
        "True + 1",
        "None",
        "1j",
        "1 < 2",
        "not 1",
        "1 and 2",
        "a.b",
        "f'{1}'",
        "1; 2",
        "x := 1",
    ],
)
def test_nothing_but_arithmetic_is_allowed(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_an_attempt_to_run_code_has_no_side_effects(tmp_path):
    marker = tmp_path / "pwned"
    with pytest.raises(CalculatorError):
        calculate(f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert not marker.exists()


def test_length_and_nesting_are_bounded():
    with pytest.raises(CalculatorError, match="longer than"):
        calculate("1+" * MAX_LENGTH + "1")
    with pytest.raises(CalculatorError):
        calculate("(" * 150 + "1" + ")" * 150)  # too deep for the parser, reported cleanly
    assert calculate("-" * 100 + "1") == 1.0  # a long but legal chain
    assert MAX_EXPONENT == 1000


def test_big_results_work_up_to_the_limit_and_are_refused_beyond_it():
    assert calculate("10 ** 100") == 1e100
    assert calculate("2 ** 900") == pytest.approx(8.452712498170644e270)
    with pytest.raises(CalculatorError, match="too large"):
        calculate("2 ** 1000")  # about 1.07e301, above the 1e300 limit
