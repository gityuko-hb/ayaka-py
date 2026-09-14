# Linear layers

Public imports live in `ayaka.layers.linear` (also exported by `ayaka.layers`).
The supplied projection implementations are integrated with the current runtime:
ReAdapter from vLLM.

| Responsibility | Module |
| --- | --- |
| Placement, borrowed DeviceContext, quantization lifecycle | `ayaka.layers.base` |
| Configuration, capability validation, LINEAR/EXPERT method selection | `ayaka.layers.quantization.base` |
| TP/EP rank views and synchronous layer collectives | `ayaka.distributed.parallel` |
| Explicit torch process-group bindings and AsyncHandle adapter | `ayaka.distributed.torch_backend` |
| Checkpoint-to-parameter binding, meta materialization, finalization | `ayaka.model_loader.module` |
| Manifest, bounded file I/O, name validation | Existing `model_loader` modules |
| Physical read ranges and logical shard descriptions | `ayaka.weights.plan`, `ayaka.weights.spec` |
| External functional GEMM registration | `ayaka.kernel.ops.custom_op` |

## Dense and quantized execution

Replicated, column, row, merged gate/up, QKV, head, low-rank, grouped and expert
projections are available. Dense linear uses PyTorch `F.linear`; no additional
GEMM backend or quantization package is installed automatically. Inference
parameters are frozen. The default return is `(output, deferred_bias)`; select
`return_bias=False` for a tensor. Deferred bias cannot be discarded.

`LinearBase` inherits `BaseLayer`. A `BaseQuantization` factory validates
placement/capabilities before allocating weights. Explicit `LinearMethodBase`
instances and qualified method names are adapted to the same lifecycle:

1. Construct the layer; its selected method creates registered storage.
2. Load every required weight/buffer, materializing meta storage if needed.
3. Call `process_weights_after_loading()` once.
4. Execute through `apply_quantization()`, which requires ready storage.

`load_module_weights()` performs steps 2 and 3. `create_weights` also receives the
parameter loader selected from the method object: a method declaring
`uses_weight_loader_v2` binds the layer's `weight_loader_v2` entry point, otherwise
`weight_loader` is used. Selection never dispatches on class names. Repacking
failures are terminal. Stateful method instances must not be reused across layers;
return a new method from the shared configuration factory. Parameter dtype and
activation dtype are separate, with equal defaults. Expert methods use the EXPERT
target and `apply_expert()`.

`register_linear_kernel()` uses the existing public OpHandle/dispatcher and
reference mode. Its default fake implementation requires dense [out, in]
storage. Kernels packing output rows must supply `output_size` or `fake_impl`.
Kernels with mutation, aliases or `out=` register directly with `custom_op`.
The default dense path does not claim custom Triton GEMM acceleration.

## Parallel groups

Supply `tp_group`, `ep_group`, `communication`, and optionally `device_context`
as constructor keyword arguments. Layer rank is `DeviceGroup.local_rank`, not
an environment/global rank. Omitted groups mean singleton execution even when
torch distributed is initialized. `disable_tp` / `disable_ep` disables the
corresponding sharding only.

`TorchCommunicationBackend({logical_group: process_group})` borrows initialized
process groups; it neither creates nor destroys them. Even WORLD must be supplied
explicitly. It validates global-rank order and group-local rank. Gather/scatter
buffers are [group.size, *local_shape] in group-rank order. The layer adapter
converts gathered tensors to last-dimension concatenation. Packed gathers preserve
projection order; row bias is added only on TP rank zero before reduction.
Asynchronous backend calls return an Ayaka AsyncHandle retaining its buffers.
The caller joins it before consuming/releasing those buffers.

`VirtualParallelContext` is an explicit test simulation. It refuses multi-rank
collectives without callbacks. Runtime groups and a parallel_context are mutually
exclusive constructor inputs.

HEAD shards divide complete query/output heads. KV_HEAD may replicate KV heads
when TP exceeds the number of KV heads. These semantics correspond to the
existing `ShardKind.HEAD` / `ShardKind.KV_HEAD` distinction. Grouped projections
accept full heads by default; pass `input_is_parallel=True` when the preceding
attention stage already produced only this TP rank's heads.

## Checkpoint binding

Architecture code supplies checkpoint-name mappings; loaders do not infer model
families or treat a global process group as TP. For separate checkpoint Q/K/V:

```python
from ayaka.layers.linear import QKVParallelLinear
from ayaka.model_loader.module import WeightBinding, load_module_weights

layer = QKVParallelLinear(
    hidden_size=16,
    head_size=4,
    total_num_heads=8,
    total_num_kv_heads=2,
    bias=False,
    return_bias=False,
)
bindings = {
    "q_proj.weight": WeightBinding("weight", "q"),
    "k_proj.weight": WeightBinding("weight", "k"),
    "v_proj.weight": WeightBinding("weight", "v"),
}
# weights is an iterable of (checkpoint_key, tensor), with all three projections.
load_module_weights(layer, weights, bindings=bindings)
```

A fused checkpoint can bind directly to `weight` without a shard ID.
Packed loaders validate physical widths using `packed_dim`, `packed_factor`
and optional `marlin_tile_size`; local tensors and consecutive local tuples are
accepted without a second TP slice. Format-specific scale/zero-point loaders
remain the quantization method's responsibility.

`load_checkpoint(module, manifest, bindings=..., selections=...)` reads a
validated CheckpointManifest using BoundedCheckpointReader. Optional TensorSlice
selections use physical checkpoint coordinates. They become contiguous or
strided FileReadPlans so row/column slices can be read without materializing the
whole tensor. Read plans and layer shard loaders must agree on the selected
local shape. Omitted selections read full checkpoint tensors. Integer packed
storage is copied verbatim; sub-byte manifest dtypes need a format-specific adapter.

Missing, unexpected, duplicate, overlapping and incomplete projection bindings
raise. Every registered parameter and persistent buffer must be covered.
Non-persistent caches are excluded. Meta materialization preserves parameter
loader attributes and tied parameter identity; dense meta tensors need an explicit
execution device. Loading validates before each parameter's writes but is not a
model-wide rollback transaction. Recreate/reload after failure. CUDA copies and
packing are ordered on the current stream; the runtime joins that stream before
publishing weights to a different execution stream.

## Scope and validation

The routed expert implementation is a correctness/reference path. It accepts
already dispatched local expert IDs and may synchronize for ID validation and
materialize per-token selected weights. Token dispatch/combine, optimized MoE
GEMM and graph-safe routed expert execution are not provided here.

The supplied model-named projection packs expose their declared projection
graphs; they do not certify checkpoint/model-family compatibility. Grouped A
projection remains dense; grouped output's quant_config applies to its B stage.

Tests cover original projection behavior, packed loader regressions, capability
and lifecycle failures, meta/tied identity, selected safetensors reads, existing
activation composition, real two-process Gloo collectives, one-GPU FP32/FP16/BF16
numerics and dense CUDA Graph replay. Virtual-rank CUDA checks do not prove
multi-GPU NCCL or end-to-end model performance.
