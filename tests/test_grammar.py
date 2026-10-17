"""CPU tests against real llguidance; fixtures adapted from the supplied Rust tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from ayaka.grammar import (
    ByteTokenTable,
    CacheLimits,
    CompileCache,
    GrammarError,
    Vocabulary,
    compile,
)
from ayaka.grammar.adapter import GBNFMaskProducer, GrammarMatcher
from ayaka.sampling.mask.producer import MaskRows


@pytest.fixture
def vocabulary():
    table = ByteTokenTable(
        tuple(bytes([i]) for i in range(256)) + (b"<eos>", b"<stop>", None),
        frozenset({256, 257}),
        lambda text: list(text.encode()),
    )
    return Vocabulary(table, 264, CacheLimits(2, 1 << 20))


def accepts(vocabulary, grammar, text):
    try:
        vocabulary.bind(grammar).advance([*text.encode(), 256])
    except GrammarError as error:
        if error.kind != "violation":
            raise
        return False
    return True


@pytest.mark.parametrize(
    "source",
    [
        "",
        "root ::= missing",
        'root ::= "\\uD800"',
        "root ::= [z-a]",
        'root ::= "x"{3,2}',
        'root ::= ("x"',
        'root ::= "x"\nroot ::= "y"',
        'other ::= "x"',
        'root ::= "\\q"',
        "root ::= []",
        'root ::= "\\U00110000"',
        'root ::= "x"{4294967296}',
        'root ::= "x" )',
    ],
)
def test_syntax(source):
    with pytest.raises(GrammarError) as error:
        compile(source)
    assert error.value.kind == "syntax"


def test_empty_language():
    with pytest.raises(GrammarError, match="language"):
        compile('root ::= loop\nloop ::= "a" loop')


@pytest.mark.parametrize(
    "source,good,bad",
    [
        (
            'root ::= [a-z]* nest\nnest ::= "(" nest ")" | "x"',
            ["abx", "x", "ab((x))", "(x)"],
            ["ab(x", "abx)", "ab", "ABx"],
        ),
        (
            'root ::= "a" nest "b" tail | "a" nest "bce"\n'
            'nest ::= "(" nest ")" | "k"\ntail ::= "cd" | "[" tail "]"',
            ["akbcd", "akbce", "a((k))b[[cd]]", "a(k)bce"],
            ["akbc", "akb", "akbcde"],
        ),
        (
            'root ::= start A a\nstart ::= "\\u000a"\n'
            'A ::= [\\x61-\\u0062]{1,2}\na ::= "\\U0001f999"',
            ["\na🦙", "\nbb🦙"],
            ["na🦙", "\nccc🦙", "\na"],
        ),
        ('root ::= ("a" |\n "b")+ [^x-z]? # comment\n', ["a", "abba!", "b\n"], ["", "abz", "xx"]),
        ("root ::= .{2,3}", ["é🦙", "\n\n", "abc"], ["a", "abcd"]),
        ('root ::= "a"{2,}', ["aa", "aaaa"], ["", "a", "aab"]),
        ('root ::= x | dead\nx ::= "x"\ndead ::= "d" dead', ["x"], ["d", "dd"]),
        (
            'root ::= scan "b"\nscan ::= "a" scan | "b" scan | ""',
            ["b", "ab", "bb", "aabb"],
            ["", "a", "ba"],
        ),
        ('root ::= ""', [""], ["x"]),
        ('root ::= "x"{0}', [""], ["x"]),
        ('root ::= dead*\ndead ::= "a" dead', [""], ["a"]),
    ],
)
def test_languages(vocabulary, source, good, bad):
    grammar = compile(source).grammar
    for text in good:
        assert accepts(vocabulary, grammar, text), text
    for text in bad:
        assert not accepts(vocabulary, grammar, text), text


def test_mask_matches_every_short_continuation(vocabulary):
    # Independent finite-language oracle includes a shorter accepting alternative
    # and overlapping recursive prefixes, exposing greedy lexer overrun bugs.
    grammar = compile('root ::= "a"* nest\nnest ::= "(" nest ")" | "x"').grammar
    language = {"a" * n + "(" * k + "x" + ")" * k for n in range(8) for k in range(8)}
    for size in range(5):
        for chars in itertools.product("ax()", repeat=size):
            text = "".join(chars)
            if not any(word.startswith(text) for word in language):
                continue
            state = vocabulary.bind(grammar)
            if text:
                state = state.advance(list(text.encode()))
            for char in "ax()":
                assert state.allows(ord(char)) == any(
                    word.startswith(text + char) for word in language
                )
            assert state.allows(256) == (text in language)


def test_prefix_cache_forcing_and_fork(vocabulary):
    grammar = compile('root ::= "prefix:ab"').grammar
    state = vocabulary.bind(grammar, list(b"prefix:"))
    assert state.position == 0
    assert state.forced(1) == (97,)
    child = state.advance([97])
    assert child.forced(5) == (98,)
    assert state.forced(1) == (97,)
    assert vocabulary.bind(grammar, list(b"prefix:")).position == 0
    assert vocabulary.cache_hits == 1
    with pytest.raises(GrammarError, match="binding"):
        vocabulary.bind(grammar, list(b"wrong"))
    with pytest.raises(GrammarError, match="binding"):
        vocabulary.bind(grammar, [256])
    with pytest.raises(ValueError):
        state.forced(0)


def test_transactional_stops_and_padding(vocabulary):
    state = vocabulary.bind(compile('root ::= "a"').grammar)
    for tokens in [[], [256], [97, 256, 97], [258], [263], [-1]]:
        with pytest.raises(GrammarError):
            state.advance(tokens)
        assert state.position == 0 and state.allows(97)
    done = state.advance([97])
    assert done.accepting() and not done.stopped()
    assert done.allows(256) and done.allows(257)
    stopped = done.advance([257])
    assert stopped.stopped() and stopped.position == 2
    with pytest.raises(GrammarError):
        stopped.advance([256])
    assert all(not done.allows(t) for t in range(258, 264))


def test_utf8_split_and_multibyte_tokens(vocabulary):
    state = vocabulary.bind(compile('root ::= "🦙"').grammar)
    data = "🦙".encode()
    for byte in data:
        assert state.allows(byte)
        assert not state.allows(256)
        state = state.advance([byte])
    assert state.allows(256)
    table = ByteTokenTable(
        (b"ab", b"a", b"b", b""),
        frozenset({3}),
        lambda s: {"ab": [0], "a": [1], "b": [2], "": []}[s],
    )
    v = Vocabulary(table, 8)
    state = v.bind(compile('root ::= "ab"').grammar)
    assert state.allows(0) and not state.allows(2)
    assert state.advance([0, 3]).stopped()
    assert state.advance([1, 2, 3]).stopped()


def test_producer_speculation_and_atomic_commit(vocabulary):
    producer = GBNFMaskProducer(vocabulary.bind(compile('root ::= "ab"').grammar))
    rows = MaskRows(np.zeros((4, 9), dtype=np.uint32), 264)
    assert producer.emit((97, 98, 256), rows) == 3
    assert producer.committed_len() == 0
    assert producer.emit((97, 99, 98), rows) == 1
    assert rows.allows(0, 97) and rows.allows(1, 98) and rows.allows(2, 256)
    with pytest.raises(ValueError):
        producer.commit((97, 99))
    assert producer.committed_len() == 0
    producer.commit((97, 98, 257))
    assert producer.committed_len() == 3


def test_rollback_and_bad_token_do_not_mutate(vocabulary):
    matcher = GrammarMatcher(vocabulary.bind(compile('root ::= "a"*').grammar))
    assert not matcher.accept_token(98)
    assert matcher.accept_token(97)
    assert matcher.accept_token(256)
    matcher.rollback(2)
    assert matcher.num_accepted() == 0 and not matcher.is_terminated()
    assert matcher.accept_token(256)
    with pytest.raises(ValueError):
        matcher.rollback(2)


def test_cache_limits_and_independent_binding(vocabulary):
    cache = CompileCache(1, 1 << 20)
    first = cache.compile('root ::= "a"')
    assert cache.compile('root ::= "a"') is first
    cache.compile('root ::= "b"')
    assert cache.compile('root ::= "a"') is not first
    no_cache = CompileCache(0, 0)
    assert no_cache.compile('root ::= "a"') is not no_cache.compile('root ::= "a"')
    for c in "abc":
        vocabulary.bind(compile(f'root ::= "{c}"').grammar)
    assert vocabulary.cached_entries == 2 and vocabulary.cached_bytes <= vocabulary.limits.bytes
    a = vocabulary.bind(first.grammar)
    b = vocabulary.bind(first.grammar)
    a.advance([97, 256])
    assert b.position == 0 and b.allows(97)


def test_no_continuation_does_not_fabricate_eos():
    table = ByteTokenTable((b"b", b""), frozenset({1}), lambda _: [0])
    with pytest.raises((GrammarError, ValueError), match="round-trip"):
        state = Vocabulary(table, 2).bind(compile('root ::= "a"').grammar)
        producer = GBNFMaskProducer(state)
        producer.emit((), MaskRows(np.zeros((1, 1), np.uint32), 2))


REASONING_JSON = (
    'root ::= "<t>" reasoning space value\n'
    'reasoning ::= [^<] reasoning | "<" open\n'
    'open ::= "/" slash | "<" open | [^/<] reasoning\n'
    'slash ::= "t" tee | "<" open | [^t<] reasoning\n'
    'tee ::= ">" | "<" open | [^><] reasoning\n'
    'value ::= object | array | string | "null"\n'
    'object ::= "{" space (string ":" space value ("," space string ":" space value)*)? space "}"\n'
    'array ::= "[" space (value ("," space value)*)? space "]"\n'
    'string ::= "\\"" [^"]* "\\""\n'
    'space ::= | " " | "\\n"\n'
)


SCANNED_CALLS = (
    'root ::= text ("<c>" nest ">")*\n'
    'text ::= | "<" open | [^<] text\n'
    'open ::= | "<" open | "c" tag | [^<c] text\n'
    'tag ::= | "<" open | [^<>] text\n'
    'nest ::= "(" nest ")" | "x"\n'
)


def test_original_reasoning_and_scanned_call_fixtures(vocabulary):
    reasoning = compile(REASONING_JSON).grammar
    for text in [
        '<t>a<b</t>{"k": [null, {"x":"y"}]}',
        "<t></t> [ ]",
        "<t>x</t>\n{ }",
        '<t><</t>{"a":{"b":{"c":[[[]]]}}}',
    ]:
        assert accepts(vocabulary, reasoning, text)
    for text in ['<t>a</t>{"k" null}', "<t>a</t>[", "<t>a{}", "<t>a</t>  {}"]:
        assert not accepts(vocabulary, reasoning, text)
    scanned = compile(SCANNED_CALLS).grammar
    for text in ["", "free <text> <c", "a<c>(x)><c>((x))>", "<<c<c x<c>x>", "<c>x>"]:
        assert accepts(vocabulary, scanned, text)
    for text in ["a<c>", "a<c>(x)> b", "<c>(x)", "a<c>(x)>><c", "<c>>"]:
        assert not accepts(vocabulary, scanned, text)


def test_unicode_empty_complement():
    with pytest.raises(GrammarError, match="language"):
        compile(r"root ::= [^\x00-\U0010ffff]")
