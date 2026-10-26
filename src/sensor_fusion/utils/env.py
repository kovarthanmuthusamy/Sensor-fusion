"""Environment report and GPU-architecture check.

Our Quadro GV100s are Volta (compute capability 7.0, `sm_70`). Recent PyTorch CUDA
builds no longer ship `sm_70` kernels; with such a build every CUDA op fails with
"no kernel image is available for execution on the device". Run this module after
any torch upgrade:

    python -m sensor_fusion.utils.env
"""

import os
import platform
import re
import sys

import torch


def is_arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Whether a torch build compiled for `arch_list` can run on a GPU of `capability`.

    Two ways a build can support a GPU:
    - native kernels (`sm_XY`) for the same major version and an equal-or-lower minor
      (an `sm_80` binary runs on an 8.6 GPU, never on 7.x or 9.x);
    - PTX (`compute_XY`) for an equal-or-older capability, which the driver
      JIT-compiles on first use (slow startup, but it works).
    """
    major, minor = capability
    target = 10 * major + minor
    for arch in arch_list:
        match = re.fullmatch(r"(sm|compute)_(\d+)[a-z]?", arch)
        if match is None:
            continue
        kind, version = match.group(1), int(match.group(2))
        if kind == "sm" and version // 10 == major and version % 10 <= minor:
            return True
        if kind == "compute" and version <= target:
            return True
    return False


def collect_env_info() -> dict[str, str]:
    """Versions and GPU details worth logging with every run."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset: all GPUs>"),
    }
    if torch.cuda.is_available():
        arch_list = torch.cuda.get_arch_list()
        info["torch_cuda"] = str(torch.version.cuda)
        info["torch_arch_list"] = " ".join(arch_list)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            supported = is_arch_supported((props.major, props.minor), arch_list)
            info[f"gpu{i}"] = (
                f"{props.name}, {props.total_memory / 2**30:.0f} GiB, "
                f"sm_{props.major}{props.minor}, "
                f"{'supported' if supported else 'NOT SUPPORTED by this torch build'}"
            )
    return info


if __name__ == "__main__":
    for key, value in collect_env_info().items():
        print(f"{key:<22}{value}")
