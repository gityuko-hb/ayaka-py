"""Execution backend capabilities, separate from attention/device support."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GraphBackendCapabilities:
    phases: tuple[str, ...] = ("decode", "prefill")
    devices: tuple[str, ...] = ("cuda",)
    output_modes: tuple[str, ...] = ("sampling_logits",)
    padding: tuple[tuple[str, str], ...] = (
        ("decode", "masked_store_zero_length_rows"),
        ("prefill", "masked_store_empty_csr_sentinel"),
    )
    capture: str = "exclusive_bootstrap_synthetic_kv"
    reset: str = "drain_then_rebuild"
    concurrency: str = "single_lane_private_flight_pools"
    scope: str = "model_forward_and_logits"


CAPABILITIES = {
    "full": GraphBackendCapabilities(),
    "breakable": GraphBackendCapabilities(scope="transformer_body_graph_eager_logits"),
    "torch_compile_piecewise": GraphBackendCapabilities(scope="eager_body_compiled_logits_graph"),
}
