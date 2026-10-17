"""Strict GBNF syntax, preserving Unicode scalar classes and rule-name case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn


class GrammarError(ValueError):
    """Grammar failure with a stable syntax/language/binding/violation category."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(f"{kind}: {message}")


@dataclass(frozen=True, slots=True)
class Expr:
    kind: str
    value: Any = None
    parts: tuple[Expr, ...] = ()


def parse(source: str) -> dict[str, Expr]:
    """Parse the uploaded Rust module's GBNF dialect; reject malformed input."""
    return Parser(source).parse()


class Parser:
    def __init__(self, source: str) -> None:
        try:
            size = len(source.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise GrammarError("syntax", "source contains a surrogate") from error
        if not 0 < size <= 8 * 1024 * 1024:
            raise GrammarError("syntax", "source must be between 1 byte and 8 MiB")
        self.source, self.at, self.nodes = source, 0, 0

    def error(self, message: str) -> NoReturn:
        offset = len(self.source[: self.at].encode("utf-8"))
        raise GrammarError("syntax", f"GBNF byte {offset}: {message}")

    def peek(self) -> str:
        return self.source[self.at : self.at + 1]

    def take(self) -> str:
        char = self.peek()
        self.at += bool(char)
        return char

    def space(self, newlines: bool) -> None:
        while self.peek():
            if self.peek() in (" \t\r\n" if newlines else " \t"):
                self.take()
            elif self.peek() == "#":
                while self.peek() and self.peek() not in "\r\n":
                    self.take()
            else:
                break

    def name(self) -> str:
        start = self.at
        while (c := self.peek()) and (c.isascii() and (c.isalnum() or c in "_-")):
            self.take()
        if start == self.at:
            self.error("expected rule name")
        return self.source[start : self.at]

    def node(self, kind: str, value: Any = None, parts: tuple[Expr, ...] = ()) -> Expr:
        self.nodes += 1
        if self.nodes > 100_000:
            self.error("grammar exceeds node limit")
        return Expr(kind, value, parts)

    def parse(self) -> dict[str, Expr]:
        rules: dict[str, Expr] = {}
        self.space(True)
        while self.peek():
            name = self.name()
            self.space(False)
            if not self.source.startswith("::=", self.at):
                self.error("expected ::=")
            self.at += 3
            self.space(True)
            body = self.alternatives(False, 0)
            if name in rules:
                self.error("duplicate rule")
            rules[name] = body
            self.space(True)
        if "root" not in rules:
            self.error("grammar has no root rule")
        pending = list(rules.values())
        while pending:
            expr = pending.pop()
            if expr.kind == "ref" and expr.value not in rules:
                self.error(f"undefined GBNF rule: {expr.value}")
            pending.extend(expr.parts)
        return rules

    def alternatives(self, nested: bool, depth: int) -> Expr:
        if depth > 128:
            self.error("grammar nesting exceeds Python limit (128)")
        options = [self.sequence(nested, depth)]
        while True:
            before = self.at
            self.space(True)
            if self.peek() != "|":
                self.at = before
                break
            self.take()
            self.space(True)
            options.append(self.sequence(nested, depth))
        return options[0] if len(options) == 1 else self.node("alt", parts=tuple(options))

    def sequence(self, nested: bool, depth: int) -> Expr:
        parts = []
        while True:
            self.space(nested)
            if not self.peek() or self.peek() in "|)\r\n":
                break
            char = self.take()
            if char == '"':
                text = []
                while self.peek() != '"':
                    if not self.peek():
                        self.error("unterminated literal")
                    text.append(self.character())
                self.take()
                part = self.node("literal", "".join(text))
            elif char == "[":
                negated = self.peek() == "^"
                if negated:
                    self.take()
                ranges = []
                while self.peek() != "]":
                    if not self.peek():
                        self.error("unterminated character class")
                    start = self.character()
                    end = start
                    if self.peek() == "-" and not self.source.startswith("-]", self.at):
                        self.take()
                        end = self.character()
                    if start > end:
                        self.error("reversed character range")
                    ranges.append((ord(start), ord(end)))
                self.take()
                if not ranges:
                    self.error("empty character class")
                part = self.node("class", (negated, tuple(ranges)))
            elif char == ".":
                part = self.node("any")
            elif char == "(":
                self.space(True)
                part = self.alternatives(True, depth + 1)
                self.space(True)
                if self.take() != ")":
                    self.error("expected closing parenthesis")
            elif char.isascii() and (char.isalnum() or char in "_-"):
                self.at -= 1
                part = self.node("ref", self.name())
            else:
                self.error("unexpected production character")
            self.space(False)
            char = self.peek()
            bounds = None
            if char and char in "*+?":
                self.take()
                bounds = {"*": (0, None), "+": (1, None), "?": (0, 1)}[char]
            elif char == "{":
                self.take()
                self.space(False)
                low = self.number()
                self.space(False)
                high = low
                if self.peek() == ",":
                    self.take()
                    self.space(False)
                    high = None if self.peek() == "}" else self.number()
                self.space(False)
                if self.take() != "}" or (high is not None and high < low):
                    self.error("invalid repetition bounds")
                bounds = (low, high)
            if bounds is not None:
                part = self.node("repeat", bounds, (part,))
            parts.append(part)
        return parts[0] if len(parts) == 1 else self.node("seq", parts=tuple(parts))

    def number(self) -> int:
        start = self.at
        while self.peek() and self.peek() in "0123456789":
            self.take()
        digits = self.source[start : self.at]
        if not digits or len(digits) > 10 or int(digits) > 0xFFFFFFFF:
            self.error("invalid repetition count")
        return int(digits)

    def character(self) -> str:
        char = self.take()
        if not char or char in "\r\n":
            self.error("unterminated character or unescaped newline")
        if char != "\\":
            return char
        escape = self.take()
        if escape in ("n", "r", "t"):
            return {"n": "\n", "r": "\r", "t": "\t"}[escape]
        if escape and escape in '"\\[]-/':
            return escape
        if escape in ("x", "u", "U"):
            count = {"x": 2, "u": 4, "U": 8}[escape]
            digits = self.source[self.at : self.at + count]
            if len(digits) != count or any(c not in "0123456789abcdefABCDEF" for c in digits):
                self.error("invalid Unicode escape")
            self.at += count
            value = int(digits, 16)
            if value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
                self.error("escape is not a Unicode scalar")
            return chr(value)
        self.error("unknown character escape")
        raise AssertionError("unreachable")
