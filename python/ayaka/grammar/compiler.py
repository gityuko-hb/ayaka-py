"""Exact GBNF lowering using one Unicode scalar per llguidance lexeme.

This deliberately uses the safe character-level strategy from the Rust oracle.
It does not port scanner extraction or lexical certification optimizations.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from time import perf_counter_ns

from .gbnf import Expr, GrammarError, parse


@dataclass(frozen=True, slots=True)
class Grammar:
    """Tokenizer-independent llguidance Lark program."""

    lark: str


@dataclass(frozen=True, slots=True)
class CompileReport:
    gbnf_bytes: int
    lark_bytes: int
    symbols: int
    earley_rules: int
    terminals: int
    character_lexemes: int
    compile_us: int
    strategy: str = "characters"


@dataclass(frozen=True, slots=True)
class Compiled:
    grammar: Grammar
    report: CompileReport


def _class_nonempty(node: Expr) -> bool:
    negated, ranges = node.value
    if not negated:
        return True
    # Count the union over Unicode scalars, excluding the surrogate interval.
    covered = 0
    for begin, end in ((0, 0xD7FF), (0xE000, 0x10FFFF)):
        cursor = begin
        for low, high in sorted(ranges):
            low, high = max(begin, low), min(end, high)
            if high >= max(cursor, low):
                covered += high - max(cursor, low) + 1
                cursor = high + 1
    return covered < 0x110000 - 0x800


def compile(gbnf: str) -> Compiled:
    """Validate GBNF, prune empty-language branches, and lower without greedy loss.

    Raises GrammarError for invalid syntax or an unproductive root language.
    No tokenizer, model, CUDA context, or native runtime is needed here.
    """
    started = perf_counter_ns()
    rules = parse(gbnf)
    # Number AST nodes by object identity: avoid recursive structural hashing.
    nodes: list[Expr] = []
    ids: dict[int, int] = {}
    pending = list(rules.values())
    while pending:
        node = pending.pop()
        if id(node) in ids:
            continue
        ids[id(node)] = len(nodes)
        nodes.append(node)
        pending.extend(node.parts)
    dependencies: list[list[int]] = []
    parents: dict[int, list[int]] = defaultdict(list)
    productive: set[int] = set()
    for i, node in enumerate(nodes):
        deps = [ids[id(p)] for p in node.parts]
        if node.kind == "ref":
            deps = [ids[id(rules[str(node.value)])]]
        dependencies.append(deps)
        for child in deps:
            parents[child].append(i)
        if (
            (node.kind in ("literal", "any") or (node.kind == "class" and _class_nonempty(node)))
            or (node.kind == "repeat" and node.value[0] == 0)
            or (node.kind == "seq" and not deps)
        ):
            productive.add(i)
    queue = deque(productive)
    remaining = [len(deps) for deps in dependencies]
    while queue:
        child = queue.popleft()
        for parent in parents[child]:
            if parent in productive:
                continue
            remaining[parent] -= 1
            if nodes[parent].kind == "alt" or remaining[parent] == 0:
                productive.add(parent)
                queue.append(parent)
    root = ids[id(rules["root"])]
    if root not in productive:
        raise GrammarError("language", "root has no finite derivation")

    def name(index: int) -> str:
        return f"r{index}"

    lines = ["%llguidance {}", f"start: {name(root)}"]
    terminals: dict[str, str] = {}

    def terminal(pattern: str) -> str:
        if pattern not in terminals:
            terminals[pattern] = f"T{len(terminals)}"
        return terminals[pattern]

    visited: set[int] = set()
    pending_ids = [root]
    while pending_ids:
        i = pending_ids.pop()
        if i in visited:
            continue
        visited.add(i)
        node = nodes[i]
        deps = [d for d in dependencies[i] if d in productive]
        pending_ids.extend(deps)
        if node.kind == "literal":
            body = " ".join(terminal(f"/[\\x{{{ord(c):x}}}]/") for c in node.value)
        elif node.kind == "class":
            negated, ranges = node.value
            pattern = "[^" if negated else "["
            for low, high in ranges:
                pattern += f"\\x{{{low:x}}}-\\x{{{high:x}}}"
            body = terminal("/" + pattern + "]/")
        elif node.kind == "any":
            body = terminal(r"/[\x{0}-\x{10ffff}]/")
        elif node.kind == "ref":
            body = name(deps[0])
        elif node.kind in ("seq", "alt"):
            body = (" | " if node.kind == "alt" else " ").join(map(name, deps))
        elif node.kind == "repeat":
            low, high = node.value
            if not deps or high == 0:
                body = ""
            else:
                suffix = {(0, None): "*", (1, None): "+", (0, 1): "?", (1, 1): ""}.get((low, high))
                if suffix is None:
                    suffix = f"{{{low},{'' if high is None else high}}}"
                body = name(deps[0]) + suffix
        else:
            raise AssertionError(node.kind)
        lines.append(f"{name(i)}: {body or chr(34) + chr(34)}")
    lines.extend(f"{label}: {pattern}" for pattern, label in terminals.items())
    lark = "\n".join(lines) + "\n"
    return Compiled(
        Grammar(lark),
        CompileReport(
            len(gbnf.encode()),
            len(lark.encode()),
            len(nodes),
            len(visited),
            len(terminals),
            len(terminals),
            (perf_counter_ns() - started) // 1000,
        ),
    )


class CompileCache:
    """Thread-safe LRU bounded by entry count and UTF-8 source/program bytes.

    The byte charge excludes Python object overhead and native runtime memory.
    """

    def __init__(self, max_entries: int = 64, max_bytes: int = 16 << 20) -> None:
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("cache limits must be nonnegative")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries: OrderedDict[str, Compiled] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def compile(self, gbnf: str) -> Compiled:
        with self._lock:
            if gbnf in self._entries:
                self._entries.move_to_end(gbnf)
                return self._entries[gbnf]
        result = compile(gbnf)
        charge = result.report.gbnf_bytes + result.report.lark_bytes
        with self._lock:
            if gbnf in self._entries:
                self._entries.move_to_end(gbnf)
                return self._entries[gbnf]
            if self.max_entries and charge <= self.max_bytes:
                self._entries[gbnf] = result
                self._bytes += charge
                while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                    _, old = self._entries.popitem(last=False)
                    self._bytes -= old.report.gbnf_bytes + old.report.lark_bytes
        return result
