"""Asking for a GPU and silently getting a CPU is the failure this file exists to exclude.

`device_refusal` is a PURE function over the facts, so every branch below is shown firing on a
box with no CUDA build and no GPU. The thin collector that reads those facts off torch is the
only part not covered here, and it is the part with no logic in it.
"""

from __future__ import annotations

import pytest

from recall.sparse import DeviceReport, SparseDeviceError, device_refusal, resolve_sparse_device

#: A GTX 1070 Ti as measured on this box: Pascal, compute capability 6.1, 8 GB.
PASCAL = (6, 1)
PASCAL_ARCHES = ("sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90")
NO_PASCAL_ARCHES = ("sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100")


def test_a_cpu_only_wheel_is_refused_by_name() -> None:
    """The exact local condition: torch 2.13.0+cpu, hardware present and unreachable.

    Reported as a WHEEL problem, not as "no GPU". Those need different fixes, and telling someone
    with a working card that they have no GPU sends them to the wrong one.
    """
    refusal = device_refusal(
        cuda_build=None, device_count=0, capability=None,
        arch_list=(), free_vram_mb=None, required_vram_mb=2048,
    )

    assert refusal is not None
    assert "CPU-only" in refusal


def test_a_cuda_wheel_with_no_visible_device_is_refused() -> None:
    refusal = device_refusal(
        cuda_build="12.8", device_count=0, capability=None,
        arch_list=("sm_80",), free_vram_mb=None, required_vram_mb=2048,
    )

    assert refusal is not None
    assert "no CUDA device" in refusal


def test_a_card_whose_architecture_the_wheel_dropped_is_refused() -> None:
    """The Pascal trap, and the reason this reads the arch list rather than assuming it.

    A wheel built without sm_61 does not politely decline. Naming the architecture and listing
    what the wheel does carry is what turns that into an actionable message.
    """
    refusal = device_refusal(
        cuda_build="12.8", device_count=1, capability=PASCAL,
        arch_list=NO_PASCAL_ARCHES, free_vram_mb=8192, required_vram_mb=2048,
    )

    assert refusal is not None
    assert "sm_61" in refusal
    assert "sm_80" in refusal


def test_insufficient_free_vram_is_refused_with_both_numbers() -> None:
    refusal = device_refusal(
        cuda_build="12.8", device_count=1, capability=PASCAL,
        arch_list=PASCAL_ARCHES, free_vram_mb=512, required_vram_mb=2048,
    )

    assert refusal is not None
    assert "512" in refusal and "2048" in refusal


def test_a_usable_card_produces_no_refusal() -> None:
    """The positive control. Without it every assertion above passes on a function that always
    refuses, which would read as a working guard while blocking every GPU run."""
    assert device_refusal(
        cuda_build="12.8", device_count=1, capability=PASCAL,
        arch_list=PASCAL_ARCHES, free_vram_mb=8192, required_vram_mb=2048,
    ) is None


def test_requesting_cuda_explicitly_raises_rather_than_falling_back(monkeypatch) -> None:
    import recall.sparse as sparse

    monkeypatch.setattr(
        sparse, "inspect_sparse_device",
        lambda requested, required_vram_mb=2048: DeviceReport(
            requested=requested, resolved="cpu", torch_cuda_build=None, device_name=None,
            capability=None, supported_architectures=(), free_vram_mb=None,
            refusal="torch is a CPU-only build",
        ),
    )

    with pytest.raises(SparseDeviceError, match="CPU-only"):
        resolve_sparse_device("cuda")


def test_auto_falls_back_without_raising(monkeypatch) -> None:
    """`auto` means "use it if it is there", so a refusal is information, not an error."""
    import recall.sparse as sparse

    monkeypatch.setattr(
        sparse, "inspect_sparse_device",
        lambda requested, required_vram_mb=2048: DeviceReport(
            requested=requested, resolved="cpu", torch_cuda_build=None, device_name=None,
            capability=None, supported_architectures=(), free_vram_mb=None,
            refusal="torch is a CPU-only build",
        ),
    )

    assert resolve_sparse_device("auto") == "cpu"


def test_requesting_cpu_never_consults_cuda() -> None:
    """Asking for CPU on a box with no torch at all must still work.

    Routed through the collector, this would import torch in order to decide it did not need
    torch.
    """
    assert resolve_sparse_device("cpu") == "cpu"
