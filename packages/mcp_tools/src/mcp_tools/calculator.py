"""A safe arithmetic evaluator.

Why not `eval`? A model chooses the expression. `eval("__import__('os').system(...)")` would run
whatever it says. Here the text is parsed into a syntax tree and only numbers and the arithmetic
operators below are allowed. Anything else (names, calls, attributes, strings) is refused.
"""

import ast
import math
import operator
from typing import Any

MAX_LENGTH = 200
MAX_EXPONENT = 1000
MAX_RESULT = 1e300

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

SUPPORTED = "numbers, + - * / // % ** and parentheses"


class CalculatorError(ValueError):
    """A problem the model can understand and fix. The message is safe to show it."""


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so True would slip through without this check.
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalculatorError(f"only numbers are allowed (supported: {SUPPORTED})")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"the exponent is too large (at most {MAX_EXPONENT})")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError:
            raise CalculatorError("cannot divide by zero") from None
        except OverflowError:
            raise CalculatorError("the result is too large") from None
    raise CalculatorError(f"that is not supported. Supported: {SUPPORTED}")


def calculate(expression: str) -> float:
    text = expression.strip()
    if not text:
        raise CalculatorError("the expression is empty")
    if len(text) > MAX_LENGTH:
        raise CalculatorError(f"the expression is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise CalculatorError(f"that is not a valid arithmetic expression ({SUPPORTED})") from None
    except (RecursionError, MemoryError):
        raise CalculatorError("the expression is too deeply nested") from None
    try:
        result = _eval(tree.body)
    except RecursionError:
        raise CalculatorError("the expression is too deeply nested") from None
    if isinstance(result, complex):  # a negative number to a fractional power
        raise CalculatorError("the result is not a real number")
    try:
        number = float(result)
    except OverflowError:
        raise CalculatorError("the result is too large") from None
    if not math.isfinite(number) or abs(number) > MAX_RESULT:  # inf - inf is nan
        raise CalculatorError("the result is too large")
    return number
