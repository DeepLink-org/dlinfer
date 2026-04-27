# Copyright (c) 2024, DeepLink. All rights reserved.
import subprocess
import time
from multiprocessing import Process

import pytest

# Maximum time (seconds) to wait for a single model test subprocess.
# Ray placement group creation has a 1800s internal timeout; set this
# higher so pytest gets a clean failure rather than an invisible hang.
DEFAULT_SUBPROCESS_TIMEOUT = 300

_DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


def _ray(args: list[str], env_extra: dict | None = None) -> None:
    import os

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    subprocess.run(["ray"] + args, env=env, **_DEVNULL)


def restart_ray_with_npu(npu: int) -> None:
    """Stop Ray and restart it with an explicit NPU resource declaration.

    On 910B, a freshly started Ray cluster reports NPU=0 in cluster_resources()
    because acl device detection races with ray.init() returning. Pre-starting
    the cluster here with --resources forces the correct NPU count so that the
    subprocess's ray.init() connects to this cluster (via lmdeploy's existing
    ValueError fallback) instead of creating a broken fresh one.

    RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 prevents Ray from
    overriding ASCEND_RT_VISIBLE_DEVICES per-worker. Without it, each worker
    gets one device (e.g., worker 1 → ASCEND_RT_VISIBLE_DEVICES=7), causing
    lmdeploy's set_device(local_rank=1) to fail since only index 0 is valid.
    """
    print(f"[ray] stopping existing cluster")
    _ray(["stop", "--force"])
    time.sleep(5)
    print(f"[ray] starting head with NPU={npu}")
    _ray(
        ["start", "--head", f"--resources={{\"NPU\": {npu}}}"],
        env_extra={"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1"},
    )
    time.sleep(3)


def cleanup_ray() -> None:
    """Force-stop all Ray processes after a test."""
    print("[ray] cleaning up cluster")
    _ray(["stop", "--force"])
    time.sleep(3)


def join_or_kill(p: Process, timeout: int = DEFAULT_SUBPROCESS_TIMEOUT) -> None:
    """Join process with timeout; kill and fail if it exceeds the limit."""
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        pytest.fail(
            f"Test subprocess did not finish within {timeout}s. "
            "Likely cause: Ray placement group waiting for NPU resources that "
            "were not registered. Check ASCEND_RT_VISIBLE_DEVICES and Ray cluster state."
        )
    if p.exitcode != 0:
        pytest.fail(f"Test subprocess exited with code {p.exitcode}.")
