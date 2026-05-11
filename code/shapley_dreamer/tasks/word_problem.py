from __future__ import annotations

import ast
import operator
import random
import re

from .base import TaskGenerator, register


_OP_FUNCS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}
_OP_AST = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}
_INT_RE = re.compile(r"-?\d+")


def _safe_eval(expr: str) -> int:
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node) -> int:
    if isinstance(node, ast.BinOp) and type(node.op) in _OP_AST:
        return _OP_AST[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    raise ValueError(f"Disallowed AST node: {ast.dump(node)}")


@register("word_problem")
class WordProblemGenerator(TaskGenerator):
    def __init__(
        self,
        seed: int = 0,
        num_operands: int = 3,
        max_value: int = 9,
        ops: tuple[str, ...] = ("+", "-"),
    ) -> None:
        if num_operands < 2:
            raise ValueError("num_operands must be >= 2")
        if max_value < 1:
            raise ValueError("max_value must be >= 1")
        unknown = set(ops) - set(_OP_FUNCS)
        if unknown:
            raise ValueError(f"Unsupported ops: {sorted(unknown)}")
        if not ops:
            raise ValueError("ops must be non-empty")
        self.num_operands = num_operands
        self.max_value = max_value
        self.ops = tuple(ops)
        self._rng = random.Random(seed)

    @classmethod
    def from_settings(cls, settings) -> "WordProblemGenerator":
        ops = tuple(o.strip() for o in settings.TASK_OPS.split(",") if o.strip())
        return cls(
            seed=settings.RANDOM_SEED,
            num_operands=settings.TASK_NUM_OPERANDS,
            max_value=settings.TASK_MAX_VALUE,
            ops=ops,
        )

    def _gen_tree(self, n_leaves: int) -> str:
        if n_leaves == 1:
            return str(self._rng.randint(0, self.max_value))
        k = self._rng.randint(1, n_leaves - 1)
        left = self._gen_tree(k)
        right = self._gen_tree(n_leaves - k)
        op = self._rng.choice(self.ops)
        return f"({left} {op} {right})"

    def sample(self) -> tuple[str, str]:
        n_leaves = self._rng.randint(2, self.num_operands)
        expr = self._gen_tree(n_leaves)
        if expr.startswith("(") and expr.endswith(")"):
            expr = expr[1:-1]
        return expr + " = ?", str(_safe_eval(expr))

    def eval(self, answer: str, gt: str) -> float:
        match = _INT_RE.search(answer)
        if match is None:
            return 0.0
        try:
            return 1.0 if int(match.group()) == int(gt) else 0.0
        except ValueError:
            return 0.0