# Python GBNF grammar integration

This ports the uploaded Grammar module's GBNF language and request-state API
into `ayaka.grammar`, against ayaka-py main at
`b3ab8959e7b6f1883c62b2eadeaaa5bc7563687d`.
The Python code continues to use llguidance's native parser and token trie,
as the Rust module did. It is not a pure-Python replacement for llguidance.

Install with `uv sync --group dev --extra cpu --extra grammar`, or substitute
`--extra cuda` for GPU use. `llguidance>=1.9.1,<2` is an optional extra; importing
the compiler does not initialize the native parser or require a model.

## Engine API

Create one Vocabulary per model/tokenizer and keep the complete model output
width, including padded logits columns. Use the tokenizer's exact byte table,
not `decode([token]).encode()`, which corrupts partial UTF-8 tokens.

```python
from ayaka.grammar import CompileCache, Vocabulary

vocabulary = Vocabulary.from_tokenizer(
    tokenizer,                     # Ayaka TokenizerLike
    projection=model_vocab_size,   # actual logits.size(-1), including padding
    stop_tokens=model_stop_ids,    # every EOS/end-of-turn alias for this model
)
compiled = CompileCache().compile('root ::= "yes" | "no"')

# After the scheduler has admitted the request with coordinator.add(...),
# and before its first plan_for(...):
coordinator.attach_grammar(request_id, vocabulary, compiled.grammar)
```

Each admitted request needs its own attachment. Install identical grammar and
vocabulary configuration on every tensor-parallel rank. The existing lifecycle
then runs `plan_for` -> `sample` -> completion publication -> `record_published`.
Sampling applies the grammar after custom operators/bias/penalties and before
temperature and filtering. Publication advances the grammar; discarded samples
and recomputation do not. `release` removes the request's matcher and planner
producer before slot reuse. Attaching a second grammar to the same request is
rejected. Attachment requires rebuilding the sampling plan.

For a grammar prefix already present in the assistant's output, pass token ids
through `attach_grammar(..., prefix=prefix_ids)`. This is only the leading part
of the grammar, never the whole conversation prompt. The prefix is validated
and consumed, but generated-token position starts at zero.

The new entry point is an **engine API**. This change does not add an HTTP/gRPC
request field or automatically convert JSON Schema response formats to GBNF.
A serving composition root can call it after admission using the model's
already-loaded tokenizer. No tokenizer/model is downloaded by this module.

## Low-level API and speculative owners

```python
from ayaka.grammar import compile
from ayaka.grammar.adapter import GBNFMaskProducer

state = vocabulary.bind(compile('root ::= "ab"').grammar)
staged = state.advance([token_a])  # parent unchanged, or GrammarError
forked = state.fork()
words = state.mask()              # tuple of packed uint32 allow bits
forced = state.forced(8)          # excludes stop tokens
producer = GBNFMaskProducer(state)
```

`GBNFMaskProducer` implements the existing MaskProducer protocol; its matcher
implements ConstraintMatcher. It can be passed to `MaskPipeline` for draft
verification. `emit(draft, rows)` fills one row per draft position plus a bonus
row, returns the valid draft-prefix length and rolls parser state back.
`commit(tokens)` advances atomically. Backend errors propagate rather than
fabricating a successful EOS for incomplete output. As in the existing producer
contract, rows beyond an invalid draft become EOS-only and must be ignored
using the returned accepted-prefix length.

`accepting()` means the grammar may end; `stopped()` means a stop token was
explicitly consumed. Multiple EOS aliases are supported and none spell content.
A finite grammar may finish its content before EOS is consumed. Empty/unusable
vocabulary slots and model-padding columns are masked. llguidance may restrict
sampling masks to its canonical forced tokenization; `advance` also validates
legal alternative token segmentations, matching the uploaded Rust API.

The coordinator integration handles **ordinary decoding**. The producer's
speculative rollback contract is tested, but this change does not wire a new
speculative scheduler, MTP or DFlash execution path. That owner should submit
its draft rows through the existing MaskPipeline and commit only accepted output.

## Compiler strategy and mapping from Rust

All emitted lexemes consume exactly one Unicode scalar, so none can greedily
overrun another lexeme. This follows the safe character-level oracle strategy
in the supplied Rust tests; it avoids approximating recursive CFGs with regexes.
Rules and AST nodes are referenced by generated names, preserving rule-name
case, Unicode literals, dot including newlines, character classes, alternation,
recursion, epsilon, comments and bounded/unbounded repetition. Empty-language
branches are pruned before rendering. Counted repetitions remain compact in the
emitted source, though native compilation still has its own resource limits.

| Uploaded Rust area | Python implementation |
| --- | --- |
| `gbnf.rs` | `grammar/gbnf.py`: parser, escapes, reference validation |
| `network.rs` | `grammar/compiler.py`: productivity worklist and reachable lowering |
| `lark.rs`, public compilation in `lib.rs` | `grammar/compiler.py`: Lark and immutable result/report |
| `cache.rs` | `CompileCache`: thread-safe count/byte-bounded LRU |
| `vocabulary.rs` | `grammar/runtime.py`: raw bytes, projection, EOS, prefix binding and LRU |
| `constraint.rs` | `GrammarConstraint`: fork/stage/advance, mask and forcing |
| `terminal.rs`, `elimination.rs`, `lexeme.rs`, `certify.rs`, `language.rs` | Not translated as optimizers; always use the exact character strategy |
| `tests.rs` | CPU tests of language fixtures, masks, transactions and live sampler integration |

This is a **functional port, not performance parity** with the Rust compiler.
Scanner extraction, state elimination, structural flattening, loop peeling and
regex-derivative certification are not reproduced. Consequently CompileReport
reports the actual character strategy and does not fabricate scanner/fuel
metrics. Large tool lattices and long free-text reasoning may need substantially
more Earley work than the optimized Rust compiler. No throughput or latency
speedup is claimed.

Input limits: 8 MiB UTF-8 source, 100,000 AST nodes, unsigned 32-bit repetition
counts, and 128 nested groups (Rust allowed 256). Native parser/lexer limits
remain enforced and reported as binding failures. Compile-cache bytes charge
UTF-8 source plus Lark; binding-cache bytes charge Lark, prefix and initial mask.
These are admission charges, not exact bounds on Python/native heap usage.
Vocabulary and mutable matchers belong to one engine thread. Bind/fork deep-copy
native state, preventing cross-request parser or lexer cache sharing.

## Sampling fixes required for the connection

Previously SamplingCoordinator discarded the planner's mask schedule and always
called the sampler with no mask. It now materializes masks and supplies MaskHandle
through the existing sampler stage. The ordinary path uses synchronous CPU mask
construction and device upload on the caller's stream, allocating per step.
It does not promise asynchronous overlap or CUDA graph capture of this CPU work.
GPU and multi-rank execution have not been validated here.

The planner now maps persistent slots to packed active rows, including reordered
and sparse batches; emits all-allow identity rows for unconstrained members of a
mixed batch; and allocates scratch windows after the entire primary block.
Empty intersections raise instead of sampling an invalid token. Grammar-active
samples use the existing TP token synchronization hook before publication.

## Verification

Focused tests use real llguidance and CPU PyTorch, not parser or torch mocks.

```bash
python -m pytest -q tests/test_grammar.py tests/test_grammar_sampling.py
ruff check python/ayaka/grammar python/ayaka/sampling/engine.py python/ayaka/sampling/plan.py tests/test_grammar.py tests/test_grammar_sampling.py
ruff format --check python/ayaka/grammar python/ayaka/sampling/engine.py python/ayaka/sampling/plan.py tests/test_grammar.py tests/test_grammar_sampling.py
pyright python/ayaka/grammar python/ayaka/sampling/engine.py python/ayaka/sampling/plan.py
```

Tests include the uploaded reasoning/JSON and scanner/call fixtures, lexical
overrun, ambiguous continuations, malformed sources, dead branches, Unicode
split across byte tokens, multiple stop tokens, prefix binding, bounded caches,
independent forks, failed staging, speculative rollback, atomic commits, actual
logit masking, discarded samples, reordered mixed batches and slot reuse.
A finite-language oracle independently checks masks over every prefix through
length four. This is not an exhaustive equivalence proof against the original
Rust optimizer, nor a CUDA/model throughput benchmark.
