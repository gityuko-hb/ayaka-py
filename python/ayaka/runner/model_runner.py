"""Minimal real model runner for the sampling vertical slice (M1–M3).

Each step re-forwards the full known-token stream of every slice that needs
output with causal SDPA attention and projects only the needed hidden rows
through the LM head (``LogitsProcessor``). This is a REAL model forward with
real weights feeding the real sampling chain and the real
executor/completion contracts — deliberately not the paged-KV runner:

  * the paged runner (KV reuse, prefix cache, chunk-boundary carry) is a
    separate milestone; the report contracts tested here do not depend on KV
    reuse. For prompt scoring the dense path recomputes the predecessor rows
    that a paged runner would carry from the previous chunk's workspace;
  * ``known_tokens`` is the scheduler-owned snapshot (prompt + published
    outputs), so recomputation is exactly the sequence the step continues;
  * slices are forwarded when they sample OR carry prompt-logprob scoring.

Deferred M3 items, with reasons pinned here so they are not rediscovered:
  * workspace/buffer reuse — the snapshot/logprob tensors ESCAPE each call
    (held by ``SampleOutputs`` until the completion boundary materializes
    them), so a runner-global reuse pool would overwrite an unpublished
    ticket's payload. Reuse belongs to the per-ticket ``ExecutionResources``
    workspace of the paged runner.
  * CUDA Graph capture — the dense recompute path has dynamic stream lengths
    per slice; capture needs the paged runner's static shapes (the sampling
    side is already graph-ready via ``coords_padded``).
  * tensor-parallel sharded vocabulary — no TP runtime is wired in this tree;
    ``LogitsProcessor`` already documents the seam.

Responsibility boundary (see ``runtime/logits.py``): the scheduler decided
which positions get scored (``BatchStepPlan.prompt_logprobs``); this runner
maps them to hidden rows; the LogitsProcessor produces logits; the
LogprobProcessor only observes; only the Sampler transforms the distribution.

Flow:
    model forward -> hidden rows -> LogitsPlan -> LogitsProcessor -> raw logits
    -> raw snapshot (FP32, raw-mode + prompt rows, before any transform)
    -> Sampler -> device tokens
    -> generation (raw + sampling modes) + prompt raw logprobs
    -> SampleOutputs (device-resident until the completion boundary).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ayaka.executor.ticket import SampleOutputs
from ayaka.logits_processor import LogitsPlan, LogitsProcessor
from ayaka.sampling.bans import BanApplier, BansProvider, LogitFillBanApplier
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sampling.logprobs import (
    LogprobEntry,
    LogprobMode,
    LogprobPlan,
    PromptLogprobSliceReport,
    SampleLogprobTensors,
    TokenIdsLogprobTensors,
    compute_prompt_logprobs,
    compute_raw_logprobs,
    compute_sampling_logprobs,
    compute_token_ids_raw_logprobs,
    compute_token_ids_sampling_logprobs,
)
from ayaka.sched.plan import PreparedStep

__all__ = ["ModelRunner"]


@dataclass(frozen=True, slots=True)
class _SamplingReportRow:
    """One generation report row that scores the SAMPLING distribution."""

    sampling_row: int
    k: int


@dataclass(frozen=True, slots=True)
class _IdsReportRow:
    """One row requesting logprobs for specific token ids."""

    sampling_row: int
    tokens: tuple[int, ...]
    mode: LogprobMode


@dataclass(frozen=True, slots=True)
class _GenerationReport:
    """Generation report rows split by mode (the two distributions are
    observed at different pipeline points: raw needs the pre-mutation
    snapshot, sampling needs the post-mutation logits after the draw)."""

    raw: tuple[LogprobEntry, ...]
    sampling: tuple[_SamplingReportRow, ...]
    ids: tuple[_IdsReportRow, ...] = ()
    ids_raw_count: int = 0


class ModelRunner:
    """Run one real model forward + LogitsProcessor + Sampler per prepared step."""

    __slots__ = ("_ban_applier", "_bans", "_coordinator", "_force_reference", "_logits", "_model")

    def __init__(
        self,
        coordinator: SamplingCoordinator,
        model,
        *,
        logit_scale: float = 1.0,
        final_logit_softcapping: float | None = None,
        force_reference: bool = False,
        bans: BansProvider | None = None,
        ban_applier: BanApplier | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._model = model
        self._logits = LogitsProcessor(
            model, logit_scale=logit_scale, final_logit_softcapping=final_logit_softcapping
        )
        self._force_reference = force_reference
        self._bans = bans
        self._ban_applier = ban_applier if ban_applier is not None else LogitFillBanApplier()

    def set_bans(self, bans: BansProvider | None) -> None:
        """Install the per-step ban provider (the engine's sampling masks)."""
        self._bans = bans

    def _apply_bans(self, prepared: PreparedStep, logits: torch.Tensor, rows: int) -> None:
        if self._bans is None:
            return
        bans = self._bans(prepared.step)
        if len(bans) != rows:
            raise ValueError(f"ban provider returned {len(bans)} sets for {rows} sampling rows")
        self._ban_applier.apply(logits, bans)

    def __call__(self, prepared: PreparedStep) -> SampleOutputs:
        self._validate_prepared(prepared)
        step = prepared.step
        plan = step.sampling
        self._coordinator.flush()

        report = self._generation_report(step)
        (
            hidden_packed,
            sampling_count,
            prompt_descriptors,
            prompt_targets,
            prompt_ks,
        ) = self._forward_prepared(prepared)

        if sampling_count != plan.num_rows:
            raise RuntimeError(
                f"model runner projected {sampling_count} sampling rows but the "
                f"sampling plan declares {plan.num_rows}"
            )
        total_rows = hidden_packed.size(0)
        logits_plan = LogitsPlan(
            projection_rows=tuple(range(total_rows)), sampling_row_count=sampling_count
        )

        token_ids: torch.Tensor
        gen_logprobs = None
        gen_rows: tuple[int, ...] = ()
        prompt_logprobs = None
        support = None
        if total_rows > 0:
            logits = self._logits(hidden_packed, logits_plan)

            # RAW-mode rows (and all prompt rows) need the pre-mutation
            # snapshot; SAMPLING-mode rows are scored from the post-mutation
            # logits after the draw, so they never enter the snapshot.
            snapshot = self._raw_snapshot(logits, report, sampling_count, prompt_descriptors)

            if plan.num_rows:
                # Sampler consumes only the sampling-row prefix of the packed
                # logits (LogitsPlan contract); prompt rows live beyond it.
                sampling_logits = logits.narrow(0, 0, plan.num_rows)
                self._apply_bans(prepared, sampling_logits, plan.num_rows)
                token_ids, support = self._coordinator.sample_with_support(
                    sampling_logits,
                    plan,
                    force_reference=self._force_reference,
                )
            else:
                token_ids = torch.empty(0, dtype=torch.long, device=logits.device)

            gen_logprobs, gen_rows = self._generation_logprobs(logits, token_ids, report, snapshot)
            ids_logprobs, ids_rows, ids_counts = self._ids_logprobs(
                logits, token_ids, report, snapshot
            )

            prompt_rows = sum(len(entry.positions) for entry in step.prompt_logprobs)
            if prompt_rows and snapshot is not None:
                prompt_logprobs = compute_prompt_logprobs(
                    snapshot[len(report.raw) + report.ids_raw_count :],
                    torch.tensor(prompt_targets, dtype=torch.long, device=logits.device),
                    prompt_ks,
                )
        else:
            token_ids = torch.empty(0, dtype=torch.long, device=hidden_packed.device)
            ids_logprobs, ids_rows, ids_counts = None, (), ()

        return SampleOutputs(
            token_ids=token_ids,
            logprobs=gen_logprobs,
            logprob_rows=gen_rows,
            prompt_logprobs=prompt_logprobs,
            prompt_logprob_slices=tuple(prompt_descriptors),
            sampling_support=support,
            token_ids_logprobs=ids_logprobs,
            ids_logprob_rows=ids_rows,
            ids_logprob_counts=ids_counts,
        )

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _generation_report(self, step) -> _GenerationReport:
        """Generation reporting rows from scheduler-owned parameters (host).

        Per sampling slice: top-k and reporting mode come from the coordinator
        host staging; rows split by mode because the two distributions are
        observed at different pipeline points.
        """
        raw: list[LogprobEntry] = []
        sampling: list[_SamplingReportRow] = []
        ids: list[_IdsReportRow] = []
        ids_raw_count = 0
        row = 0
        for scheduled in step.slices:
            if not scheduled.sample_last_query:
                continue
            k = self._coordinator.logprob_k_for(scheduled.request_id)
            if k is not None:
                if self._coordinator.logprob_mode_for(scheduled.request_id) is LogprobMode.RAW:
                    raw.append(LogprobEntry(sampling_row=row, k=k))
                else:
                    sampling.append(_SamplingReportRow(sampling_row=row, k=k))
            ids_tokens = self._coordinator.token_ids_logprobs_for(scheduled.request_id)
            if ids_tokens:
                mode = self._coordinator.logprob_mode_for(scheduled.request_id)
                ids.append(_IdsReportRow(sampling_row=row, tokens=ids_tokens, mode=mode))
                if mode is LogprobMode.RAW:
                    ids_raw_count += 1
            row += 1
        return _GenerationReport(
            raw=tuple(raw), sampling=tuple(sampling), ids=tuple(ids), ids_raw_count=ids_raw_count
        )

    def _validate_prepared(self, prepared: PreparedStep) -> None:
        """Validate host contracts before even flushing sampling device updates."""
        prepared.validate()

    def _forward_prepared(self, prepared: PreparedStep):
        """Forward hook for runners consuming execution-lease KV addresses."""
        return self._forward_rows(prepared.step)

    def _forward_rows(self, step):
        """Forward every slice that samples or scores; gather its needed rows.

        Returns (packed hidden ``[R, hidden]``, sampling-row count, prompt
        descriptors, prompt targets, prompt ks). Packed order: sampling rows
        first (packed sampling order), then prompt-scored rows ordered by
        (slice_index, position) — matching the descriptors' contract.
        """
        device = next(self._model.parameters()).device
        prompt_entries = {entry.slice_index: entry for entry in step.prompt_logprobs}

        sampling_rows: list[torch.Tensor] = []
        scoring_rows: list[torch.Tensor] = []
        descriptors: list[PromptLogprobSliceReport] = []
        targets: list[int] = []
        ks: list[int] = []
        for index, (scheduled, value) in enumerate(zip(step.slices, step.inputs, strict=True)):
            entry = prompt_entries.get(index)
            if not scheduled.sample_last_query and entry is None:
                continue
            stream = value.known_tokens[: scheduled.query_end]
            tokens = torch.tensor(stream, dtype=torch.long, device=device)
            positions = torch.arange(len(stream), dtype=torch.long, device=device)
            attention = self._dense_attention()
            with torch.no_grad():
                hidden = self._model.forward_hidden(tokens, positions, attention)
            assert isinstance(hidden, torch.Tensor)
            if scheduled.sample_last_query:
                sampling_rows.append(hidden[-1])
            if entry is not None:
                for position in entry.positions:
                    scoring_rows.append(hidden[position - 1])
                    targets.append(stream[position])
                    ks.append(entry.k)
                descriptors.append(
                    PromptLogprobSliceReport(
                        slice_index=index,
                        start=scheduled.query_start,
                        end=scheduled.query_end,
                        scored_positions=entry.positions,
                    )
                )
        return (
            torch.stack([*sampling_rows, *scoring_rows]),
            len(sampling_rows),
            descriptors,
            targets,
            ks,
        )

    def _dense_attention(self):
        """Causal SDPA callback over the full recomputed stream (oracle shape)."""
        scale = self._model.config.scaling

        def callback(layer_index: int, query, key, value):
            q = query.transpose(0, 1).unsqueeze(0)
            k = key.transpose(0, 1).unsqueeze(0)
            v = value.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
            return out.squeeze(0).transpose(0, 1)

        return callback

    def _raw_snapshot(
        self,
        logits: torch.Tensor,
        report: _GenerationReport,
        sampling_count: int,
        prompt_descriptors: list[PromptLogprobSliceReport],
    ):
        """FP32 copy of the RAW-mode report rows' logits, before any transform.

        SAMPLING-mode rows never enter the snapshot: their distribution is
        read from the post-mutation logits after the draw. Order: raw
        generation rows first (their sampling-row indices), then raw-ids
        rows (same snapshot, scored for specific tokens), then prompt rows
        in (slice, position) order — matching the compute helpers.
        """
        gen_rows = [entry.sampling_row for entry in report.raw]
        gen_rows.extend(row.sampling_row for row in report.ids if row.mode is LogprobMode.RAW)
        prompt_rows: list[int] = []
        cursor = sampling_count
        for prompt_report in prompt_descriptors:
            for _ in prompt_report.scored_positions:
                prompt_rows.append(cursor)
                cursor += 1
        if not gen_rows and not prompt_rows:
            return None
        rows = torch.tensor([*gen_rows, *prompt_rows], dtype=torch.long, device=logits.device)
        return logits.index_select(0, rows).to(torch.float32)

    def _ids_logprobs(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        report: _GenerationReport,
        snapshot,
    ) -> tuple[TokenIdsLogprobTensors | None, tuple[int, ...], tuple[int, ...]]:
        """Logprob cho token cụ thể theo mode của từng row.

        RAW rows đọc snapshot (sau các raw top-k rows); SAMPLING rows đọc
        post-mutation packed logits. Trả (payload, rows sorted, counts).
        """
        if not report.ids:
            return None, (), ()
        md = self._coordinator.md
        parts: list[tuple[int, tuple[int, ...], TokenIdsLogprobTensors, int]] = []
        raw_index = 0
        for entry in report.ids:
            if entry.mode is LogprobMode.RAW:
                assert snapshot is not None
                offset = len(report.raw) + raw_index
                z = snapshot[offset]
                tensors = compute_token_ids_raw_logprobs(
                    z.unsqueeze(0),
                    (entry.tokens,),
                    token_ids[entry.sampling_row : entry.sampling_row + 1],
                )
                raw_index += 1
            else:
                processed = logits[entry.sampling_row : entry.sampling_row + 1]
                tensors = compute_token_ids_sampling_logprobs(
                    processed,
                    (entry.tokens,),
                    token_ids[entry.sampling_row : entry.sampling_row + 1],
                    md.active("temperature").index_select(
                        0, torch.tensor([entry.sampling_row], device=logits.device)
                    ),
                    md.active("top_k").index_select(
                        0, torch.tensor([entry.sampling_row], device=logits.device)
                    ),
                    md.active("top_p").index_select(
                        0, torch.tensor([entry.sampling_row], device=logits.device)
                    ),
                    md.active("min_p").index_select(
                        0, torch.tensor([entry.sampling_row], device=logits.device)
                    ),
                )
            parts.append((entry.sampling_row, entry.tokens, tensors, 0))

        parts.sort(key=lambda part: part[0])
        count = len(parts)
        width = max(len(part[1]) for part in parts)
        device = logits.device
        merged = torch.full((count, width), float("-inf"), dtype=torch.float32, device=device)
        merged_selected = torch.empty(count, dtype=torch.float32, device=device)
        for index, (_row, tokens, tensors, _idx) in enumerate(parts):
            length = len(tokens)
            if length:
                merged[index, :length] = tensors.logprobs[0, :length]
            merged_selected[index] = tensors.token_logprob[0]
        return (
            TokenIdsLogprobTensors(logprobs=merged, token_logprob=merged_selected),
            tuple(part[0] for part in parts),
            tuple(len(part[1]) for part in parts),
        )

    def _generation_logprobs(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        report: _GenerationReport,
        snapshot,
    ) -> tuple[SampleLogprobTensors | None, tuple[int, ...]]:
        """Merge raw + sampling mode results into one report payload.

        Rows are merged by sampling_row; the top width is the max across
        modes, padded with -1/-inf so materialization stays uniform. Raw rows
        read the pre-mutation snapshot; sampling rows read the post-mutation
        packed logits (the sampler already applied penalties/mask in place).
        """
        if not report.raw and not report.sampling:
            return None, ()
        parts: list[tuple[int, str, SampleLogprobTensors, int]] = []
        if report.raw:
            assert snapshot is not None, "raw-mode rows require the pre-mutation snapshot"
            tensors = compute_raw_logprobs(
                snapshot[: len(report.raw)], token_ids, LogprobPlan(entries=report.raw)
            )
            for index, entry in enumerate(report.raw):
                parts.append((entry.sampling_row, "raw", tensors, index))
        if report.sampling:
            rows = torch.tensor(
                [row.sampling_row for row in report.sampling],
                dtype=torch.long,
                device=logits.device,
            )
            md = self._coordinator.md
            processed = logits.index_select(0, rows)
            tensors = compute_sampling_logprobs(
                processed,
                token_ids.index_select(0, rows),
                [row.k for row in report.sampling],
                md.active("temperature").index_select(0, rows),
                md.active("top_k").index_select(0, rows),
                md.active("top_p").index_select(0, rows),
                md.active("min_p").index_select(0, rows),
            )
            for index, row in enumerate(report.sampling):
                parts.append((row.sampling_row, "sampling", tensors, index))

        parts.sort(key=lambda part: part[0])
        count = len(parts)
        top_width = max(part[2].top_token_ids.size(1) for part in parts)
        device = logits.device
        merged_lp = torch.empty(count, dtype=torch.float32, device=device)
        merged_ids = torch.full((count, top_width), -1, dtype=torch.long, device=device)
        merged_vals = torch.full(
            (count, top_width), float("-inf"), dtype=torch.float32, device=device
        )
        for mode in ("raw", "sampling"):
            positions = torch.tensor(
                [slot for slot, part in enumerate(parts) if part[1] == mode],
                dtype=torch.long,
                device=device,
            )
            if positions.numel() == 0:
                continue
            first = next(part for part in parts if part[1] == mode)
            width = first[2].top_token_ids.size(1)
            source_lp = torch.stack(
                [part[2].token_logprob[part[3]] for part in parts if part[1] == mode]
            )
            merged_lp.index_copy_(0, positions, source_lp)
            if width and top_width:
                source_ids = torch.stack(
                    [part[2].top_token_ids[part[3]] for part in parts if part[1] == mode]
                )
                source_vals = torch.stack(
                    [part[2].top_logprobs[part[3]] for part in parts if part[1] == mode]
                )
                merged_ids[:, :width].index_copy_(0, positions, source_ids)
                merged_vals[:, :width].index_copy_(0, positions, source_vals)
        rows_out = tuple(part[0] for part in parts)
        return (
            SampleLogprobTensors(
                token_logprob=merged_lp, top_token_ids=merged_ids, top_logprobs=merged_vals
            ),
            rows_out,
        )
