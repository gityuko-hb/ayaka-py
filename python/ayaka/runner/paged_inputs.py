"""Host validation against initialization-time paged attention bindings.

This checks addressing, not ownership. The runtime must separately validate the
lease, storage incarnation and allocator generations before enqueueing any work.
No tensor reads, device synchronization or manager queries belong here.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from ayaka.attention.spec import AttentionGroupSpec
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec
from ayaka.memory.views import ExecutionMemoryView, SequenceExecutionView
from ayaka.sched.plan import PreparedStep
from ayaka.utils.math_utils import div_ceil


@dataclass(frozen=True, slots=True)
class PagedGroupBinding:
    group: AttentionGroupSpec
    storage: BaseKVStorageSpec


def validate_paged_inputs(
    prepared: PreparedStep,
    bindings: Mapping[str, PagedGroupBinding],
    max_model_len: int,
) -> None:
    """Reject incompatible read coverage/write addresses before metadata launch.

    Call after ``PreparedStep.validate``. All groups are checked before the
    caller builds even the first group's device metadata.
    """
    memory = prepared.memory_view
    if isinstance(memory, ExecutionMemoryView) and len(bindings) != 1:
        raise ValueError("homogeneous execution view requires exactly one KV group")
    write_slots: dict[str, set[int]] = {name: set() for name in bindings}
    for scheduled, view in zip(prepared.step.slices, memory.sequences, strict=True):
        if scheduled.query_end > max_model_len:
            raise ValueError("scheduled range exceeds model context")
        if not isinstance(view, SequenceExecutionView):
            names = [g.group_name for g in view.groups]
            if len(set(names)) != len(names) or set(names) != set(bindings):
                raise ValueError("lease groups disagree with runner bindings")
        for name, binding in bindings.items():
            group, storage = binding.group, binding.storage
            page_size = group.page_size
            width = div_ceil(scheduled.query_end, page_size)
            window = group.spec.sliding_window
            read_start = 0 if window is None else max(0, scheduled.query_start + 1 - window)
            if isinstance(view, SequenceExecutionView):
                if not isinstance(memory, ExecutionMemoryView) or memory.page_size != page_size:
                    raise ValueError("lease page size disagrees with runner binding")
                logical_blocks = tuple(range(len(view.block_table)))
                table, slots = view.block_table, view.write_slots
                padding_page, padding_slot = memory.padding_page, memory.padding_slot
            else:
                selected = next(g for g in view.groups if g.group_name == name)
                if (
                    selected.layer_ids != group.layer_ids
                    or selected.page_size != page_size
                    or selected.storage_kind != storage.kind.value
                    or selected.dtype != storage.dtype
                ):
                    raise ValueError("lease group geometry/storage/dtype disagrees with binding")
                if (
                    selected.attention_token_start != read_start
                    or selected.attention_token_stop != scheduled.query_end
                    or selected.retained_token_start
                    != (0 if window is None else max(0, scheduled.query_end - window))
                    or selected.retained_token_stop != scheduled.query_end
                ):
                    raise ValueError("lease attention coverage disagrees with scheduled range")
                logical_blocks, table = selected.logical_blocks, selected.block_table
                slots = selected.write_slots
                padding_page, padding_slot = selected.padding_page, selected.padding_slot
            if (
                not 0 <= padding_page < storage.capacity_pages
                or padding_slot != padding_page * page_size
            ):
                raise ValueError("invalid lease padding address")
            if len(logical_blocks) != len(table) or len(set(logical_blocks)) != len(table):
                raise ValueError("lease block table must have unique logical blocks")
            if any(type(block) is not int or not 0 <= block < width for block in logical_blocks):
                raise ValueError("lease logical block lies outside scheduled context")
            if any(
                type(page) is not int
                or not 0 <= page < storage.capacity_pages
                or page == padding_page
                for page in table
            ):
                raise ValueError("lease block table has an invalid physical page")
            if len(set(table)) != len(table):
                raise ValueError("logical blocks must not alias a physical page")
            pages = dict(zip(logical_blocks, table, strict=True))
            if any(block not in pages for block in range(read_start // page_size, width)):
                raise ValueError("lease block table does not cover cached attention context")
            for slot in slots:
                logical, offset = divmod(slot.logical_position, page_size)
                if (
                    slot.physical_page != pages.get(logical)
                    or slot.page_offset != offset
                    or slot.flat_slot != slot.physical_page * page_size + offset
                    or getattr(slot, "logical_block", logical) != logical
                ):
                    raise ValueError("lease write slot disagrees with its block table")
                if slot.flat_slot in write_slots[name]:
                    raise ValueError("requests must not share a physical KV write slot")
                write_slots[name].add(slot.flat_slot)
