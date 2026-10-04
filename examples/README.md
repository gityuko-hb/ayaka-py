# Ví dụ chạy Qwen3 với Ayaka

Thư mục này có hai script chạy checkpoint Qwen3 cục bộ qua API Python của Ayaka:

| Script | Mục đích |
| --- | --- |
| [chat_qwen3.py](chat_qwen3.py) | Chat tương tác hoặc chạy một prompt, stream câu trả lời và hiển thị thời gian, thông lượng. |
| [compare_transformers.py](compare_transformers.py) | Chạy lần lượt Transformers và Ayaka trên cùng checkpoint, prompt và thiết bị để xem các chỉ số sinh văn bản. |

Cả hai khởi tạo runtime trực tiếp trong tiến trình; không cần chạy HTTP server.

## Chuẩn bị

Chạy các lệnh từ thư mục gốc của repository, với Python 3.13, `uv` và `rtk` theo
[hướng dẫn của repository](../AGENTS.md) và [RTK.md](../RTK.md).

Cài môi trường cho GPU NVIDIA:

```powershell
rtk proxy uv sync --group dev --extra cuda
```

Hoặc cài môi trường CPU:

```powershell
rtk proxy uv sync --group dev --extra cpu
```

Chọn một extra phù hợp: `cpu` và `cuda` loại trừ nhau. Các lệnh `uv run --no-sync` bên dưới
dùng môi trường đã cài ở bước này.

Chuẩn bị checkpoint Qwen3 dạng Hugging Face safetensors, chẳng hạn `Qwen/Qwen3-0.6B`.
Thư mục checkpoint cần có:

- `config.json`;
- trọng số `model.safetensors`, hoặc các shard kèm `model.safetensors.index.json`;
- các tệp tokenizer tương ứng, gồm `tokenizer.json` và `tokenizer_config.json`.

Truyền đường dẫn thư mục này qua `--model-path`. Các script không có bước tải checkpoint
tự động. Nếu bỏ tùy chọn này, cả hai dùng một snapshot cố định trong cache người dùng:

```text
~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

Các lệnh mẫu dưới đây giả sử checkpoint nằm tại `./models/Qwen3-0.6B`; thay bằng đường dẫn
thực tế trên máy, đặt trong dấu ngoặc kép nếu có khoảng trắng.

## Chat tương tác

```powershell
rtk proxy uv run --no-sync python examples/chat_qwen3.py --model-path "./models/Qwen3-0.6B"
```

Nhập tin nhắn sau dấu nhắc `You >`. Script lưu lịch sử trong phiên chạy và đưa toàn bộ
lịch sử vào prompt cho lượt tiếp theo. Sau mỗi câu trả lời, script in các chỉ số token,
độ trễ và thông lượng.

| Lệnh trong phiên chat | Tác dụng |
| --- | --- |
| `/clear`, `/reset` | Xóa lịch sử hội thoại, giữ lại system prompt. |
| `/exit`, `/quit` | Thoát phiên chat. |
| `/help` | Hiển thị các lệnh hỗ trợ. |

Lịch sử không được lưu xuống đĩa và không tự cắt ngắn khi vượt giới hạn ngữ cảnh.
Dùng `/clear` để bắt đầu lại, hoặc tăng `--max-model-len` nếu bộ nhớ cho phép.
Giới hạn ngữ cảnh phải đủ cho cả prompt (gồm lịch sử và các token ChatML) lẫn số token
đầu ra yêu cầu.

### Chạy một prompt

```powershell
rtk proxy uv run --no-sync python examples/chat_qwen3.py --model-path "./models/Qwen3-0.6B" --prompt "Viết hàm Python tính Fibonacci." --max-tokens 128
```

Khi có `--prompt`, script sinh một câu trả lời, in chỉ số rồi thoát.

Chạy trên CPU với sampling greedy:

```powershell
rtk proxy uv run --no-sync python examples/chat_qwen3.py --model-path "./models/Qwen3-0.6B" --device cpu --prompt "Giải thích KV cache trong hai câu." --temperature 0 --max-tokens 64
```

### Tùy chọn của `chat_qwen3.py`

| Tùy chọn | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--model-path` | Snapshot cache nêu trên | Thư mục chứa checkpoint và tokenizer. |
| `--prompt` | Không có | Chạy một lượt thay vì mở chat tương tác. |
| `--system-prompt` | `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` | Chỉ dẫn hệ thống cho cả hai chế độ. |
| `--device` | `auto` | `auto`, `cuda` hoặc `cpu`. |
| `--max-tokens` | `512` | Số token đầu ra tối đa mỗi lượt. |
| `--max-model-len` | `2048` | Giới hạn ngữ cảnh dùng khi khởi tạo runtime và KV cache. |
| `--temperature` | `0.7` | Nhiệt độ sampling; `0` chọn greedy. |
| `--top-p` | `0.8` | Ngưỡng nucleus sampling. |

Xem trợ giúp đầy đủ:

```powershell
rtk proxy uv run --no-sync python examples/chat_qwen3.py --help
```

## So sánh Transformers và Ayaka

```powershell
rtk proxy uv run --no-sync python examples/compare_transformers.py --model-path "./models/Qwen3-0.6B" --prompt "Viết một câu danh ngôn ngắn về cuộc sống." --max-tokens 40 --device cuda
```

Script chạy Transformers trước, giải phóng model, rồi chạy Ayaka. Hai lượt dùng cùng
prompt ChatML, thiết bị và dtype, với `temperature=0.7`, `top_p=0.8`.
Kết quả gồm văn bản sinh ra và bảng chỉ số trên terminal; script không có tùy chọn xuất
báo cáo Markdown hoặc JSON.

| Tùy chọn | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--model-path` | Snapshot cache nêu trên | Thư mục checkpoint Qwen3. |
| `--prompt` | `Viet mot cau danh ngon ngan ve cuoc song.` | Nội dung prompt dùng cho cả hai lượt. |
| `--max-tokens` | `40` | Số token đầu ra tối đa. |
| `--device` | `auto` | `auto`, `cuda` hoặc `cpu`. |

Phía Ayaka có giới hạn ngữ cảnh cố định `1024` token. Khi chạy CUDA, script yêu cầu decode
CUDA graphs với các bucket `(1, 2, 4)`; trên CPU, decode graphs bị tắt. Script hiện không có
cờ CLI để thay đổi các thiết lập này.

```powershell
rtk proxy uv run --no-sync python examples/compare_transformers.py --help
```

## Thiết bị và cách đọc chỉ số

Với `--device auto`, cả hai script chọn CUDA nếu `torch.cuda.is_available()` trả về `True`,
nếu không sẽ chọn CPU. Phía Ayaka dùng:

| Thiết bị | Backend runtime | Dtype |
| --- | --- | --- |
| CUDA | `triton` | `bfloat16` nếu được hỗ trợ, nếu không là `float16`. |
| CPU | `reference` | `float32`. |

`auto` không tự chuyển sang CPU nếu khởi tạo Triton hoặc runtime CUDA thất bại.
Để kiểm tra đường chạy CPU, chọn `--device cpu` rõ ràng.

Các script tính chỉ số từ đồng hồ phía Python:

| Chỉ số | Cách tính trong script |
| --- | --- |
| TTFT | Thời gian từ mốc bắt đầu đo đến sự kiện đầu tiên được ghi nhận. Với Ayaka, đó là `ContentDelta` đầu tiên. |
| Prefill throughput (chỉ ở chat) | Số prompt token / TTFT. Đây là giá trị ước lượng, không phải thời gian kernel prefill riêng. |
| Decode speed | Số completion token / thời gian từ sự kiện đầu tiên đến khi kết thúc generation. |
| Overall throughput | (Prompt token + completion token) / tổng thời gian generation được đo. |
| Total latency | Thời gian của lượt generation được đo; không gồm tải model và khởi tạo runtime. |

Với Ayaka, mốc đo bắt đầu sau `pipeline.prepare`, nên không bao gồm bước chuẩn bị request.
Thời gian đo còn bao gồm xử lý stream và ghi ra terminal.

### Giới hạn của phép so sánh

`compare_transformers.py` là ví dụ đo một request mỗi engine. Khi đọc kết quả, cần lưu ý:

- `TimedStreamer.put()` phía Transformers ghi nhận cả lần nhận prompt ban đầu. Vì vậy
  TTFT hiện có thể phản ánh lúc prompt được đưa vào streamer, trước khi sinh token đầu
  tiên; không thể so sánh trực tiếp với TTFT từ `ContentDelta` của Ayaka.
- Script không có vòng warmup, lặp nhiều lần hoặc cố định seed. Hai engine có thể sinh
  văn bản và số completion token khác nhau, dù cùng giới hạn `--max-tokens`.
- Các dòng về continuous batching, prefix caching và lợi thế kiến trúc trong bảng in ra
  là mô tả cố định của script. Lượt chạy này không đo concurrency, cache hit hay mức tăng
  tốc của các tính năng đó.

Khi lưu kết quả để so sánh, ghi kèm checkpoint, GPU/CPU, phiên bản thư viện, backend,
dtype, prompt và số completion token thực tế. Thư mục hiện không có script benchmark
quét toàn bộ chiến lược của Ayaka.

## Xử lý lỗi thường gặp

- **Không tìm thấy model hoặc `config.json`:** kiểm tra `--model-path` trỏ đến thư mục
  checkpoint đầy đủ, không phải thư mục `snapshots` cha hoặc tên model trên Hub.
- **Lỗi import `ayaka`, PyTorch hoặc Transformers:** chạy lại lệnh `uv sync` với extra phù
  hợp, rồi chạy script từ thư mục gốc bằng `uv run --no-sync`.
- **Lỗi CUDA/Triton:** kiểm tra môi trường đã cài extra `cuda` và PyTorch nhận GPU; có thể
  chạy riêng với `--device cpu` để kiểm tra đường chạy reference.
- **Vượt ngữ cảnh hoặc thiếu bộ nhớ:** giảm `--max-tokens`, rút ngắn prompt hoặc xóa lịch
  sử chat. Với script chat, chỉ tăng `--max-model-len` khi đủ bộ nhớ cho KV cache.
