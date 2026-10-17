# Ayaka

Ayaka là LLM inference engine viết bằng Python, sử dụng PyTorch và Triton. Engine kết hợp
continuous batching, paged KV cache, tái sử dụng prefix và CUDA graphs để thực thi các mô hình
decoder-only. Có thể sử dụng qua thư viện Python hoặc HTTP server với streaming output.

README này ánh xạ các tính năng tới implementation trong [python/ayaka/](python/ayaka/).
Các đường dẫn đều tương đối với repository để có thể đọc tiếp mã nguồn ngay trên GitHub.

## Tính năng và mã nguồn

| Tính năng | Nội dung | Mã nguồn chính |
| --- | --- | --- |
| **Native models** | Implementation cho QWen, Qwen2, Qwen3, LLaMA, GPT-2 và Phi; cấu hình, forward và mapping checkpoint riêng cho từng kiến trúc. | [models/](python/ayaka/models/), [CLI chọn model](python/ayaka/commands/cli.py) |
| **Checkpoint loading** | Đọc safetensors, kiểm tra header và tensor, lập kế hoạch đọc có giới hạn bộ nhớ, bind trọng số vào module và xác nhận weight readiness. Có module giải quyết nguồn local/Hugging Face Hub. | [model_loader/](python/ayaka/model_loader/), [weights/](python/ayaka/weights/) |
| **Continuous batching** | Lập lịch theo từng iteration, ưu tiên decode, chia prefill thành chunk và ghép prefill/decode trong batch. Có ngân sách token, admission, aging và hook preemption. | [continuous.py](python/ayaka/sched/continuous.py), [policy.py](python/ayaka/sched/policy.py), [budget.py](python/ayaka/sched/budget.py) |
| **Paged KV cache** | Quản lý KV theo page, page table và nhóm attention; tách storage, layout, materialization và retention policy. | [kvcache/](python/ayaka/kvcache/), [paged_runner.py](python/ayaka/runner/paged_runner.py) |
| **Prefix caching** | Tra cứu prefix bằng radix tree, tái sử dụng các page tương thích, quản lý ownership và resume state để giảm phần prompt cần tính lại. | [prefix/](python/ayaka/prefix/), [grouped_prefix.py](python/ayaka/kvcache/grouped_prefix.py) |
| **Memory budgets và host tiering** | Theo dõi capacity, allocation, workspace và transaction; serving có thể bật host KV tier với ngân sách riêng. | [memory/](python/ayaka/memory/), [tiering.py](python/ayaka/memory/tiering.py), [ServingRuntime](python/ayaka/runtime/serving.py) |
| **Runtime cache resize** | Kiểm tra ngân sách trước khi rebuild KV cache; serving từ chối resize khi còn request đang hoạt động. Có API xem trạng thái và thay đổi số page. | [resize.py](python/ayaka/kvcache/resize.py), [runtime/serving.py](python/ayaka/runtime/serving.py) |
| **Attention backends** | Contract chung cho metadata và KV binding; có Triton, PyTorch reference, FlashAttention và FlashInfer. Backend kiểm tra capability trước khi thực thi. | [attention/](python/ayaka/attention/), [backend/](python/ayaka/attention/backend/), [forward_context.py](python/ayaka/attention/forward_context.py) |
| **CUDA graphs** | Capture/replay decode theo batch bucket; prefill graphs bật qua cấu hình. Có pool, kiểm tra generation/binding, thống kê và đường chạy eager khi phù hợp. | [runner/graph/](python/ayaka/runner/graph/), [decode_cuda_graph_runner.py](python/ayaka/runner/decode_cuda_graph_runner.py), [prefill_cuda_graph_runner.py](python/ayaka/runner/prefill_cuda_graph_runner.py) |
| **Thực thi theo phase và PDMux** | Chính sách riêng cho prefill/decode; PDMux tách hai phase thành các execution lane, với tài nguyên và giới hạn ticket được kiểm tra khi khởi tạo. | [configs/phase.py](python/ayaka/configs/phase.py), [worker/multiplexer.py](python/ayaka/worker/multiplexer.py), [execution_lane.py](python/ayaka/worker/execution_lane.py) |
| **Sampling** | Greedy, temperature, top-k, top-p, min-p; repetition/frequency/presence penalties, logit bias, seed, nhiều output và báo cáo logprobs. | [params.py](python/ayaka/sampling/params.py), [sampling/ops/](python/ayaka/sampling/ops/), [rng.py](python/ayaka/sampling/rng.py) |
| **Structured output** | JSON/JSON Schema qua XGrammar; bitmask pipeline và trie matcher cho constrained decoding ở tầng sampling. | [serving/constraints.py](python/ayaka/serving/constraints.py), [sampling/mask/](python/ayaka/sampling/mask/), [xgrammar_matcher.py](python/ayaka/sampling/custom/xgrammar_matcher.py) |
| **Speculative decoding** | N-gram proposer tích hợp scheduler, target verification và commit/rollback tài nguyên. Có greedy/rejection acceptance và policy điều chỉnh độ dài draft. | [speculative/](python/ayaka/speculative/), [coordinator.py](python/ayaka/speculative/coordinator.py), [mode.py](python/ayaka/speculative/mode.py) |
| **FP8 linear layers** | Các recipe per-tensor, rowwise, block, W8A16 và MXFP8; loader xử lý scale và sharding, kernel triển khai GEMM/GEMV tương ứng. | [layers/linear/fp8.py](python/ayaka/layers/linear/fp8.py), [model_loader/fp8.py](python/ayaka/model_loader/fp8.py), [kernel/triton/gemm/](python/ayaka/kernel/triton/gemm/) |
| **Streaming và request lifecycle** | Façade sync/async, output stream, cancellation, deadline và bounded admission; completion coordinator quản lý settlement và thu hồi tài nguyên. | [runtime/llm.py](python/ayaka/runtime/llm.py), [request/](python/ayaka/request/), [executor/completion.py](python/ayaka/executor/completion.py) |
| **HTTP serving** | Endpoint theo định dạng OpenAI/Anthropic, streaming SSE, API key, reasoning parser và tool-call parser. | [serving/http.py](python/ayaka/serving/http.py), [protocol.py](python/ayaka/serving/protocol.py), [parsers.py](python/ayaka/serving/parsers.py) |
| **Metrics và diagnostics** | Prometheus metrics, trạng thái scheduler/graph, thống kê request và SLO; báo cáo capability/profile theo phần cứng và ngân sách. | [serving/stats.py](python/ayaka/serving/stats.py), [metrics/](python/ayaka/metrics/), [device/](python/ayaka/device/) |
| **Distributed building blocks** | Process group, điều phối KV, distributed step/completion, DP routing và worker theo rank; collective dispatch có verified-P2P probe. | [distributed/](python/ayaka/distributed/), [runtime/dp.py](python/ayaka/runtime/dp.py), [worker/distributed.py](python/ayaka/worker/distributed.py), [device_comm/](python/ayaka/device_comm/) |

### Phạm vi hỗ trợ hiện tại

- **CLI và `ServingRuntime` phục vụ trên một device.** Các module distributed, routing và
  migration có API riêng; CLI `serve` chưa cung cấp tuỳ chọn khởi chạy TP/PP/DP.
- **CPU dùng backend `reference`.** Triton và CUDA graphs cần GPU CUDA cùng các dependency
  tương ứng. FlashAttention/FlashInfer là extras tuỳ chọn; `ServingRuntime` nhận các backend
  này, còn CLI hiện nhận `triton` hoặc `reference`.
- **Speculative decoding tích hợp scheduler hỗ trợ n-gram.** `draft_model`, `mtp` và `eagle3`
  được đánh dấu `UNAVAILABLE` trong [mode.py](python/ayaka/speculative/mode.py). Đường draft-model
  reference riêng nằm ở [runner/speculative_runner.py](python/ayaka/runner/speculative_runner.py).
  Tree verification, graph verification, guided decoding và tensor parallelism có gate riêng;
  không mặc định áp dụng cho n-gram.
- **Host tiering là đường được serving chấp nhận.** Cấu hình NVMe/swap bị từ chối tại
  [runtime/serving.py](python/ayaka/runtime/serving.py).
- **Kernel MoE và LoRA chưa nối vào runtime.** [kernel/triton/moe/](python/ayaka/kernel/triton/moe/)
  và [kernel/triton/lora/](python/ayaka/kernel/triton/lora/) có implementation nhưng chưa có runtime
  consumer. Các kernel [comm/](python/ayaka/kernel/triton/comm/) và
  [p2p_signal.py](python/ayaka/kernel/triton/p2p_signal.py) được `device_comm` sử dụng.
- Danh sách model ở trên là các **kiến trúc native đã triển khai**. Checkpoint phải đáp ứng
  validation của từng model; FP8, attention và graph support còn phụ thuộc recipe, shape,
  backend và phần cứng.

## Bản đồ codebase

Luồng phục vụ chính đi từ HTTP/CLI tới runtime, scheduler, worker/executor rồi runner và
model/attention/kernel. Scheduler quyết định thứ tự logic; memory và KV cache quyết định khả
năng cấp phát; completion xác nhận kết quả trước khi trả tài nguyên.

| Đường dẫn | Trách nhiệm |
| --- | --- |
| [commands/](python/ayaka/commands/) | Entry point `ayaka serve` và lựa chọn native model. |
| [serving/](python/ayaka/serving/) | HTTP, protocol, chuẩn bị prompt, generation, router và migration. |
| [runtime/](python/ayaka/runtime/) | Engine loop, façade LLM, serving assembly và IPC client/core. |
| [configs/](python/ayaka/configs/) | Cấu hình model, memory, cache, scheduler, parallel và execution phase. |
| [sched/](python/ayaka/sched/) | Admission, batching, policy, preemption và immutable step plan. |
| [executor/](python/ayaka/executor/), [worker/](python/ayaka/worker/) | Execution ticket, flight lifecycle, fence, lane và completion. |
| [runner/](python/ayaka/runner/) | Chuẩn bị input, paged forward, sampling và graph execution. |
| [models/](python/ayaka/models/), [layers/](python/ayaka/layers/) | Native model và các layer embedding, linear, norm, activation, RoPE. |
| [model_loader/](python/ayaka/model_loader/), [weights/](python/ayaka/weights/) | Nguồn checkpoint, read plan, tensor mapping và weight readiness. |
| [memory/](python/ayaka/memory/), [kvcache/](python/ayaka/kvcache/), [prefix/](python/ayaka/prefix/) | Allocation, KV storage/tiering/retention và prefix reuse. |
| [attention/](python/ayaka/attention/), [sampling/](python/ayaka/sampling/) | Backend contract, forward context, sampler, penalties và constraints. |
| [speculative/](python/ayaka/speculative/) | Proposer, policy, verifier, acceptance và speculative transaction. |
| [generation/](python/ayaka/generation/README.md) | Bản chuyển Rust: staged generation, reasoning budget, suppression, draft state và adapter cho paged runner. |
| [device/](python/ayaka/device/), [device_comm/](python/ayaka/device_comm/), [distributed/](python/ayaka/distributed/) | Capability phần cứng, collective policy và phối hợp nhiều rank/replica. |
| [kernel/triton/](python/ayaka/kernel/triton/), [kernel/csrc/](python/ayaka/kernel/csrc/) | GPU kernels, PyTorch reference kernels và native IPC extension. |
| [tokenizers/](python/ayaka/tokenizers/), [utils/](python/ayaka/utils/) | Tokenization/detokenization, stop checking và helper dùng chung. |

Đọc sâu hơn: [thiết kế scheduler](python/ayaka/sched/README.md) và
[linear layers](python/ayaka/layers/linear/README.md).

## Cài đặt

Yêu cầu **Python 3.13** và **uv**. Chạy lệnh từ repository root. Các lệnh dùng RTK theo
[RTK.md](RTK.md); cấu hình dependency và extras nằm trong [pyproject.toml](pyproject.toml).

```bash
# Môi trường phát triển với CPU PyTorch
rtk proxy uv sync --group dev --extra cpu

# Hoặc môi trường GPU CUDA
rtk proxy uv sync --group dev --extra cuda
```

Extras `cpu` và `cuda` loại trừ nhau. Extra `cuda` chọn Triton trên Linux hoặc
`triton-windows` trên Windows. Có thể bổ sung `--extra flashinfer` hoặc `--extra flash-attn`
khi dùng backend tương ứng qua thư viện Python.

## Chạy HTTP server

CLI nhận **thư mục checkpoint local**, chứa `config.json`, safetensors và tokenizer phù hợp.
Thay `./models/checkpoint` bằng đường dẫn checkpoint của bạn; dùng `--tokenizer` nếu tokenizer
nằm ở thư mục khác.

```bash
# GPU CUDA + Triton
rtk proxy uv run --no-sync python -m ayaka serve ./models/checkpoint --device cuda --backend triton

# CPU reference sau khi cài extra cpu
rtk proxy uv run --no-sync python -m ayaka serve ./models/checkpoint --device cpu --backend reference

# Xem toàn bộ tuỳ chọn
rtk proxy uv run --no-sync python -m ayaka serve --help
```

Server mặc định lắng nghe tại `http://127.0.0.1:8000`, với tên model `ayaka`.
Các tuỳ chọn thường dùng:

| Tuỳ chọn | Mục đích |
| --- | --- |
| `--served-model-name`, `--host`, `--port` | Tên model trả qua API và địa chỉ server. |
| `--kv-pages`, `--page-size`, `--max-model-len` | Capacity KV và giới hạn context phục vụ. |
| `--max-requests`, `--batch-tokens`, `--prefill-chunk` | Giới hạn sequence và ngân sách batching. |
| `--max-admitted-requests`, `--max-concurrent-requests` | Giới hạn admission và request ở frontend. |
| `--decode-graph`, `--graph-buckets`, `--graph-pool-bytes` | Capture decode graphs và ngân sách graph. |
| `--execution-config` | File JSON chính sách thực thi theo phase; prefill graphs là opt-in. |
| `--api-key` | Danh sách key phân cách bằng dấu phẩy; mặc định đọc `AYAKA_API_KEY`. |
| `--reasoning-parser think`, `--tool-parser hermes` | Bật parser reasoning hoặc tool-call phù hợp model. |

Capacity KV phải đủ context được phục vụ và padding page. Điều chỉnh `--max-model-len`
và `--kv-pages` theo model/phần cứng; thông số mặc định không phù hợp mọi checkpoint.

Các endpoint được khai báo tại [serving/http.py](python/ayaka/serving/http.py):

| Endpoint | Nội dung |
| --- | --- |
| `POST /v1/chat/completions`, `POST /v1/completions` | Chat/text generation theo định dạng OpenAI. |
| `POST /v1/responses` | Generation theo định dạng Responses. |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Messages và đếm token theo định dạng Anthropic. |
| `GET /v1/models` | Model được phục vụ. |
| `GET /health`, `GET /healthz`, `GET /ping`, `GET /readyz` | Liveness và readiness. |
| `GET /metrics`, `GET /stats` | Prometheus metrics khi bật expose metrics; thống kê runtime/SLO. |
| `GET /v1/cache/status`, `POST /v1/cache/resize` | Quản trị cache khi app được gắn admin interface. |

Chi tiết các trường request, validation và chuyển đổi protocol nằm trong
[protocol.py](python/ayaka/serving/protocol.py) và [prepare.py](python/ayaka/serving/prepare.py).

## Sử dụng qua thư viện

- [ServingRuntime](python/ayaka/runtime/serving.py) lắp ráp serving từ **model đã load** và
  đường dẫn tokenizer; hỗ trợ cấu hình backend, memory, host tiering và execution phase.
- [AyakaLLM / AsyncAyakaLLM](python/ayaka/runtime/llm.py) cung cấp API sync/async trên một
  `Engine`, nhận [Request](python/ayaka/request/schema.py) và trả stream
  [OutputEvent](python/ayaka/request/stream.py).
- [ServingConfig](python/ayaka/configs/serving.py),
  [SamplingParams](python/ayaka/sampling/params.py) và
  [SpeculativeDecodingConfig](python/ayaka/speculative/config.py) là các điểm vào cấu hình
  serving, sampling và n-gram speculation.
- [Generation](python/ayaka/generation/README.md) cung cấp API điều khiển độc lập với device.
  `Request.reasoning_budget` và `suppressed_token_ids` đã được nối vào paged runner;
  state machine MTP/DFlash cần numerical backend trước khi bật trong serving.

## Phát triển và kiểm tra

Chạy `uv sync` với extra phù hợp trước các lệnh `--no-sync`:

```bash
rtk proxy uv run --no-sync pytest -q
rtk proxy uv run --no-sync ruff check .
rtk proxy uv run --no-sync ruff format --check .
rtk proxy uv run --no-sync pyright .
rtk proxy uv build
```

Pytest được cấu hình tìm test trong `tests/`, với timeout 30 giây và marker `gpu`/`slow`.
`tests/*` hiện bị Git ignore; khi bổ sung test cần kiểm tra file đã được track. Kết quả CPU
reference cần được phân biệt với validation trên GPU CUDA thực tế.

Quy ước đóng góp nằm trong [AGENTS.md](AGENTS.md); sử dụng
[pull request template](.github/PULL_REQUEST_TEMPLATE.md) để ghi rationale và kết quả kiểm tra.
Giấy phép: [LICENSE](LICENSE).
