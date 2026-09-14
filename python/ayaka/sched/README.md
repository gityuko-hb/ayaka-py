# Scheduler

`ayaka.sched` turns admitted requests into one immutable `BatchStepPlan` per
engine iteration. It owns *logical* ordering and per-step decisions only:
physical KV/pages remain authoritative in `StepRuntime.prepare`, which may
accept, reject, or force the scheduler to reshape a candidate step.

## Module map

| Responsibility | Module |
| --- | --- |
| Abstract scheduler contract (`add_request`, `abort`, `schedule`, `update_from_output`) | `ayaka.sched.base` |
| Policy-free lifecycle machinery: admission, ownership, abort settlement, reports | `ayaka.sched.core` |
| Runtime/admission/preemption protocols and scheduler errors | `ayaka.sched.interfaces` |
| Reference scheduler: whole-prompt prefill, one-query decode | `ayaka.sched.eager` |
| Production continuous batching with chunked prefill and preemption | `ayaka.sched.continuous` |
| Implementation selection by name | `ayaka.sched.factory` |
| Token/sequence budgets and online time estimation | `ayaka.sched.budget` |
| Waiting-queue ordering and bounded-starvation aging | `ayaka.sched.policy` |
| Preemption adapters and deterministic victim selection | `ayaka.sched.preemption` |
| Advisory physical admission prechecks | `ayaka.sched.admission` |
| Observations, finish reasons and per-step reports | `ayaka.sched.outcome` |
| Immutable host contracts (`ScheduledSlice`, `BatchStepPlan`, `PreparedStep`, ...) | `ayaka.sched.plan` |
| Validated settings and capabilities | `ayaka.configs.scheduler` |

## Implementations

`create_scheduler(kind, ...)` picks one implementation; `eager`/`reference`/`debug`
map to `EagerScheduler`, `continuous`/`production` map to `ContinuousScheduler`.

`EagerScheduler` is the deterministic correctness/debug baseline: waiting
requests are considered in policy order, a prefill slice covers the whole
remaining prompt, and decode schedules one query per running request. It has no
chunking, mixed batches, cache-affinity ranking, bypasses, or preemption.

`ContinuousScheduler` is the production path. It implements decode-first
continuous batching with round-robin decode fairness, chunked prefill, mixed
PREFILL+DECODE steps, head-of-line bypass, bounded-starvation aging, optional
prefix/cache-affinity ranking, advisory admission hooks, transient-pressure
shrink before preemption, delegated recompute/swap preemption, and rollback-safe
queue mutation.

Both keep **one execution ticket in flight at a time**. CPU/GPU overlap, PP
micro-batches, and speculative multi-step execution are execution-pipeline
features and belong outside this synchronous core.

## Per-step pipeline

1. `add_request(request)` validates length feasibility via the resolved plan,
   checks sampling/stop support, enforces request/queue ceilings, creates a
   lifecycle and sequence handle, and enqueues an ADMITTED report.
2. `schedule()` builds candidate `BatchStepPlan`s against a `BatchBudget`, then
   asks `StepRuntime.prepare` to validate the plan against current KV and
   physical memory. `StepPrepareError` may shrink the candidate transiently,
   preempt one victim, or fail one request.
3. `StepRuntime.adopt` freezes the step into an `ExecutionTicket`. Only after
   prepare/adopt succeed does logical queue ownership move.
4. `update_from_output(CompletionResult)` settles the ticket, re-queues partial
   prefills, promotes completed prefills to running, and handles ignored or
   aborted requests.
5. `flush_reports()` applies accumulated `RequestReport`s to the
   `LifecycleManager` as one immutable `SchedulerReport` (queue counts,
   preemption counts). Engine wiring should call it after each
   `update_from_output`.

```python
scheduler = create_scheduler(
    "continuous",
    plan,              # ResolvedSchedulerPlan
    lifecycle_manager, # LifecycleManager
    runtime,           # StepRuntime: prepare / adopt / cancel
    allocator,         # SequenceAllocator: create / release / advance_epoch
    admission=admission,      # optional AdmissionAdvisor
    preemption=preemption,    # optional PreemptionController
    prefix_hints=hints,       # optional PrefixHintProvider
)

scheduler.add_request(request)
while scheduler.has_unfinished:
    ticket = scheduler.schedule()
    if ticket is None:
        break
    result = run_and_settle(ticket)   # executor + CompletionCoordinator
    scheduler.update_from_output(result)
    scheduler.flush_reports()
```

## Scheduling decisions

`BatchBudget.from_plan` bounds each iteration by the smaller of
`max_num_scheduled_tokens` and `execution.compute.max_num_batched_tokens`, plus
the sequence cap. `physical_token_slots` applies the runner's padding multiple,
and `largest_fittable` binary-searches the largest prefill chunk that fits.

`policy.rank_waiting` supports `fcfs`, `priority`, `lpm`/`longest_prefix`/
`cache_affinity`, `lof`/`longest_output`, and `routing_key`; `order()` layers
bounded-starvation aging on top. Policy only ranks eligible queue entries. It
never allocates KV, mutates lifecycle state, or treats a prefix hint as
ownership.

Preemption is delegated: `select_preemption_victim` deterministically picks the
lowest-priority, newest request, and `DisabledPreemptionController` /
`CallbackPreemptionController` adapt the physical mode (`recompute` or `swap`).
Admission advisors (`AlwaysAdmit`, `CallbackAdmissionAdvisor`,
`CompositeAdmissionAdvisor`) are advisory only; `prepare` remains the oracle.

## Reports and abort semantics

`RequestOutcome` values are observations, not states; the lifecycle manager owns
the transition mapping. Compute reports (`PREFILL_CHUNK`, `PREFILL_DONE`,
`DECODED`) must carry the completed `ScheduledSlice`, the committed KV version,
and exactly the sampled IDs the slice requested, so a count-only report cannot
manufacture output tokens.

`abort(request_id)` is safe at any lifecycle point, including mid-step. For an
adopted mixed batch, only that request's cancellation token is marked; the
ticket is cancelled only once every request in it is pending abort, and
sequence release/reporting is deferred until the ticket settles.

## Invariants and current scope

- `ScheduledSlice` carries request epoch and expected KV state version;
  `RequestStepInput.validate_slice` rejects stale identity, out-of-range bounds,
  and sampling off the known-token boundary.
- `BatchStepPlan` is frozen; `sampling_rows` must exactly match slice sampling
  boundaries, and one request has at most one slice per step.
- Without a sampling coordinator (`sampling=` on the scheduler) the core stays
  greedy-only and rejects non-greedy requests with `UNSUPPORTED_SAMPLING`. With
  one, `SamplingPlan` flags and the packed active-row map come from
  `ayaka.sampling.engine.SamplingCoordinator`, and the executor backend samples
  through `ayaka.runtime.sampling_runner`. Parallel sampling (`n > 1`) is still
  rejected; text stop strings need a tokenizer-backed output owner.
- `configs.scheduler` resolves and cross-checks capacities, padding, chunking,
  and workspace limits before any allocation, and the current baseline rejects
  execution/preemption combinations it cannot certify.

## Validation

Host-only tests exercise the real lifecycle with fake device owners; no model or
physical KV is involved.

```bash
uv run --no-sync pytest -q tests/test_continuous_scheduler.py
uv run --no-sync pytest -q tests/test_s1_scheduler_host.py
uv run --no-sync pytest -q tests/test_scheduler_config.py
```

`tests/*` is currently ignored by git; verify new tests are tracked before
submitting.
