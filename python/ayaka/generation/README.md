# Generation: bản chuyển từ Rust

Module này chuyển phần điều khiển generation từ 12 file Rust được cung cấp
sang Python. `ayaka.generation` không import Torch, không cấp phát KV và không
thực thi model. Các giá trị operation mô tả công việc mà backend phải thực hiện.

## Phần đã nối vào runtime Ayaka

- `Request.reasoning_budget`: giới hạn reasoning bằng các chuỗi token mở/đóng.
  `PagedModelRunner` áp mask trước sampling, kể cả sampling ở cuối prefill.
- `Request.suppressed_token_ids`: loại token khỏi mọi lần chọn của paged runner.
- `GenerationControls` cập nhật state từ `RequestLifecycle.output_token_ids`.
  Replay, forward thất bại và token chưa commit không tiêu budget. Gọi lại cùng
  một snapshot không tăng bộ đếm. State được xoá khi request kết thúc.
- `accept_greedy` dùng chung `accept_prefix`. Tham số `stops=` mới của oracle
  giữ stop token làm token do target chọn. Lời gọi cũ giữ nguyên kết quả.
- `ConstraintProducer` chuyển một staged `Constraint` sang `MaskProducer`, dùng
  được với `SamplingPlanner.attach`. Preview ghi mask nhưng không commit draft.
- `ResidentKVEngine` từ chối control nếu runner chưa khai báo hỗ trợ.
- Hai lane PDMux chia sẻ cùng `GenerationControls`; cuối prefill và decode đều
  áp cùng một state. Legacy dense speculative runner từ chối các control mới
  trước admission vì đường đó không áp mask cho mọi lookahead row.

Runtime hiện tại vẫn dùng `RequestLifecycle`, completion coordinator và KV owner
để quản lý tiến độ/commit. Không chạy thêm một `Generation` cạnh mỗi lifecycle:
hai bộ đếm cho cùng một request sẽ tạo ra hai nguồn sự thật.

Ví dụ với engine có `PagedModelRunner`:

```python
from ayaka.generation import ReasoningBudget
from ayaka.request.schema import Request, RequestId, StopCriteria
from ayaka.sampling.params import SamplingParams

# Thay các ID ví dụ bằng token ID thực của tokenizer/template đang phục vụ.
request = Request(
    RequestId("reasoning-demo"),
    prompt_token_ids=(1, 2),
    sampling=SamplingParams(temperature=0.0),
    stop=StopCriteria(max_tokens=128),
    reasoning_budget=ReasoningBudget(
        tokens=32,
        start=(10,),
        end=(11, 12),
        open=True,  # Prompt đã mở reasoning; đếm từ token được sinh đầu tiên.
    ),
    suppressed_token_ids=(7,),
)
lifecycle = engine.submit(request)
```

Đây là trường của Python Request IR. PR này chưa thêm các trường đó vào payload
OpenAI/Anthropic HTTP hoặc `GenerationSpec`. Native `ignore_eos`, stop strings,
`min_tokens` và output text tiếp tục theo `ResolvedStopPolicy` hiện có.

Request có reasoning/suppression dùng decode thường: verifier hiện tại chưa áp
mask control riêng cho từng extension row, nên `request_disable_reason` trả
`GRAMMAR` và không tạo draft. Không đổi thuật toán sampled verification hay RNG
đang dùng cho request không có các control mới.

## Ánh xạ file

| Rust được cung cấp | Python |
| --- | --- |
| `lib(1).rs` | `generation.py`, `__init__.py` |
| `acceptance.rs` | `acceptance.py` |
| `controls.rs` | `controls.py` |
| `draft_budget.rs` | `draft_budget.py` |
| `draft_control.rs` | `draft_control.py` |
| `draft_transition.rs` | `draft_transition.py` |
| `method.rs` | `method.py` |
| `plain.rs` | `plain.py` |
| `mtp.rs` | `mtp.py` |
| `dflash.rs` | `dflash.py` |
| `round.rs` | `round.py` |
| `shaping.rs` | `shaping.py` |

`contracts.py` thay các import `magnitude_executor`/`magnitude_artifacts` không
được đính kèm; `adapter.py` chứa các điểm nối Ayaka. Token ID là `int`, danh sách
token là `tuple`, lỗi kiểm tra là `ValueError`/`TypeError`, descriptor dùng frozen
dataclass. `DraftPolicy` là enum; budget cố định nằm trên factory (`with_budget`).

## Giao thức Generation và backend

```python
from ayaka.generation import Generation, InputLayout, Options, StartedRound

generation = Generation(
    prompt=(1, 2),
    layout=InputLayout(count=2),
    options=Options(
        max_tokens=8,
        output_capacity=8,
        context_limit=64,
        vocabulary=32,
    ),
)
generation.resume_at()  # Chỉ sau khi backend đã cài numerical state ở vị trí 0.
bound = generation.publication_bound(16)  # Reserve output credit trước khi chạy.
started = generation.start_round("request-1", allowance=16)
assert isinstance(started, StartedRound)
operation = started.round_forward().into_operation("request-1", generation.resident_position)

# Backend thực thi operation và trả một sample cho prefill kết thúc prompt.
samples = (4,)  # Chỉ là kết quả ví dụ; đây không phải một model forward.
transition = started.prepare_round_transition("request-1", samples)

# Backend phải reconcile KV theo transition.decision.accepted_rows ở ĐÂY.
generation, effects = started.commit(transition)
output = generation.take(bound)
```

Một target round giữ quyền mutate generation cho đến commit/cancel/fail/evict.
`prepare_round_transition` stage grammar, method, token và output trên bản riêng.
Transition gắn owner/round/version, không thể commit lại hoặc dùng cho request
khác. Drop transition không thay đổi logical state. Chỉ commit sau khi numerical
owner reconcile xong các input rows; generation không chứng minh thay backend
rằng KV đã hoàn tất.

`MethodWork.operations` và `MethodEffects.operations` là head transactions.
Backend chạy chúng, gọi `prepare_method_transition(operation, HeadOutcome(...))`,
reconcile entry rows, rồi `commit_method_transition`. Sau đó gọi `start_round`
lại. `FeatureReader` trả owned host `FeatureRows`; `FeatureRef.boundary` phải được
backend giữ sống độc lập, checkpoint chỉ giữ weak reference cho device boundary.

Các bất biến chính:

- Verification chấp nhận prefix draft bằng target samples và thêm một target
  correction/bonus. Stop token không được tính là draft match.
- `resident_position` đếm input rows đã commit; `accepted_position` có thể đi
  trước một token. `start_reconciliation` nhập token đó mà không chọn/phát token
  mới, kể cả khi generation đã kết thúc.
- Eviction replay đến numerical boundary trước eviction; không sample lại
  token đã chấp nhận hoặc nhập nhầm successor trước decode kế tiếp.
- `usage` tính cả EOS/stop token, output queue không chứa token stop. Chỉ số
  output liên tục sau `take` và `discard_output`; queue đầy sẽ chặn round mới.
- Checkpoint drafter bỏ successor token ngoài prefix, giữ lại feature đã chọn
  nó. MTP đọc target history không giữ các feature/token của history riêng.
- Forced runs, proposal width và publication đều bị chặn bởi output/context/
  max token budget. History shaping giữ 64 token sinh gần nhất như mã Rust.

## MTP/DFlash và giới hạn kiểm chứng

Đã chuyển state machine của MTP own-history, MTP đọc target history và DFlash
block, cùng checkpoint, feature pairs, maintenance transactions và controller
chọn hai độ rộng generated/verified. Các phần này chạy được bằng API backend ở
trên và có kiểm thử feature-reader/head-outcome trên CPU.

Chưa có neural MTP/DFlash backend phục vụ thật trong repo này. PR không đánh dấu
các mode đó là production, không nạp draft weights, và không giả lập tốc độ GPU.
Để bật chúng cần backend thực hiện hidden feature capture, priming, chained/block
head forward, KV reconcile/rollback và lifetime của boundary feature. Không thể
suy ra các kernel và model đó chỉ từ 12 file host được cung cấp.

Các kiểu ngoài crate cũng là giới hạn của đối chiếu: layout Python khai báo rõ
span atomic/any, host features dùng bytes với `row_bytes` cố định, selection dùng
descriptor chứ không tự sinh số ngẫu nhiên. `draft_ladder` của dependency Rust
không được cung cấp: bản Python chọn powers of two cộng width tối đa. Những phần
này là contract thích nghi với Python, không tuyên bố tương đương ABI hoặc bit
với các dependency Rust chưa có mã nguồn.

Hai thay đổi có chủ đích so với mã Rust:

1. Forced reasoning closure vẫn giao với grammar; tổ hợp bất khả thi báo lỗi
   trước khi commit, không mở lại token grammar đã cấm.
2. Adaptive acceptance learning chỉ publish vào `DraftLoad` khi transition đã
   commit. Preview bị bỏ không làm thay đổi quan sát của request khác.

Reasoning budget không tự ánh xạ từ “reasoning effort”. Start/end là chuỗi ID do
caller cung cấp. Nếu model đã phát một phần end tag lúc budget hết, chỉ phần còn
thiếu được force. `max_tokens`/context vẫn có quyền kết thúc trước khi đủ end tag.

## Kiểm tra

```bash
uv sync --group dev --extra cpu
uv run --no-sync pytest -q
uv run --no-sync ruff check python/ayaka/generation \
  python/ayaka/request/schema.py python/ayaka/runner/paged_runner.py \
  python/ayaka/runtime/resident.py python/ayaka/speculative/resources.py \
  python/ayaka/speculative/acceptance/greedy.py
uv run --no-sync ruff format --check .
uv run --no-sync pyright .
```

Nếu có RTK, thêm `rtk proxy` như `AGENTS.md` yêu cầu; môi trường thực hiện bản
chuyển này không có executable `rtk`.

`tests/test_generation*.py` bao gồm transcript từ test Rust, property test
prefix acceptance, prepare/drop/retry, stale transition, output backpressure,
stop/length/context, replay/resume, checkpoint không pin arena, adaptive learning
và phân phối rows. Test Qwen3 CPU chạy model thật nhỏ, scheduler, paged attention,
sampler, KV commit và publication trong một batch gồm request thường/control,
cả đường một runner lẫn PDMux hai lane (CPU thực thi serialized).

Kiểm thử này không chạy CUDA, không biên dịch toàn crate Rust và không chứng minh
hiệu năng hoặc parity số học của neural MTP/DFlash trên GPU. `ruff check .` còn
90 lỗi ở các file không đổi trong `kernel/triton/moe/`; `pyright .` báo 38 lỗi ở
`examples/` và subtree MoE đó. Các file của PR qua lint và typecheck riêng.
