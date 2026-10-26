import pytest
import torch

from sensor_fusion.utils.env import collect_env_info, is_arch_supported

VOLTA = (7, 0)
# Illustrative arch lists: an older build that still ships Volta, and a newer one that dropped it.
WITH_VOLTA = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
WITHOUT_VOLTA = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


def test_volta_runs_on_build_with_sm70() -> None:
    assert is_arch_supported(VOLTA, WITH_VOLTA)


def test_volta_rejected_by_build_without_sm70() -> None:
    assert not is_arch_supported(VOLTA, WITHOUT_VOLTA)


def test_newer_minor_runs_older_cubin_of_same_major() -> None:
    assert is_arch_supported((8, 6), ["sm_80"])


def test_cubin_never_runs_on_older_minor() -> None:
    assert not is_arch_supported((8, 0), ["sm_86"])


def test_older_ptx_is_jit_compilable() -> None:
    assert is_arch_supported(VOLTA, ["sm_80", "compute_70"])


def test_env_info_works_without_gpu() -> None:
    info = collect_env_info()
    assert {"python", "torch", "cuda_available"} <= info.keys()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_installed_torch_runs_on_local_gpu() -> None:
    assert is_arch_supported(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
    # An arch list can be wrong about what the driver accepts; a real kernel launch can't.
    x = torch.randn(64, 64, device="cuda")
    assert torch.isfinite(x @ x).all()
