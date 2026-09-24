"""Plain-torch oracles for the Triton kernels.

Every function here mirrors the numerics of a registered custom op and is
passed as ``reference=`` to :func:`ayaka.kernel.ops.custom_op`. Keeping them
out of the kernel modules means ``AYAKA_FORCE_REFERENCE_OPS`` and
``verify_against_reference`` can exercise them on any device, and the kernel
modules stay launch code plus kernels. These modules must not import kernels:
the dependency only runs one way, kernel module -> reference.
"""
