# Ascend Ray NPU Resource Patch

本文记录一个可选的 Ray 初始化 patch。当前该 patch 默认不启用；如果在
Ascend A2 等环境中遇到 Ray 无法为 LMDeploy placement group 分配 `NPU`
资源的问题，可以尝试临时启用该 patch 进行验证。

## 背景

在 LMDeploy PyTorchEngine 使用 Ray 创建 worker placement group 时，Ascend
设备会被映射为 Ray 自定义资源 `NPU`：

```python
placement_group_specs = [{"NPU": 1.0} for _ in range(world_size)]
```

如果 Ray 本地集群启动时没有注册 `NPU` 资源，`ray.cluster_resources()` 中就
不会包含足够的 `NPU`。此时 placement group 会一直等待，最终可能报错：

```text
INFO tpu.py:571: Failed to auto-detect TPU type.
ValueError: Cannot provide a placement group of placement_group_specs=[
{'NPU': 1.0, 'node:10.201.20.35': 0.001},
{'NPU': 1.0}, {'NPU': 1.0}, {'NPU': 1.0}
] within 1800 seconds.
See `ray status` to make sure the cluster has enough resources.
```

其中 `Failed to auto-detect TPU type` 本身不是 Ascend NPU 的根因，但它经常和
Ray 资源探测日志一起出现。真正需要确认的是 `ray status` 或
`ray.cluster_resources()` 中是否缺少 `NPU` 资源。

## 适用场景

可以考虑尝试该 patch 的情况：

- 在 Ascend A2 等机器上使用 LMDeploy + Ray 启动本地推理任务。
- 报错中出现 `Cannot provide a placement group`。
- `placement_group_specs` 中请求了 `{'NPU': 1.0}`。
- `ray status` 中没有看到 `NPU`，或 `NPU` 数量小于 `world_size`。

当前 Ascend A3 环境未复现该 Ray 资源问题，因此该 patch 暂不默认启用。

## Patch 代码

将下面函数加入 `dlinfer/dlinfer/framework/lmdeploy_ext/device/__init__.py`：

```python
def patch_ray_init():
    """Monkey-patch lmdeploy's init_ray_cluster to register custom NPU resources.

    Ray does not auto-detect Ascend NPUs; without registering custom resources
    at ray.init() time, placement groups requesting ``{'NPU': 1}`` never schedule
    on a fresh local cluster.
    """
    import os
    import logging
    import lmdeploy.pytorch.ray as _ray_mod

    logger = logging.getLogger("dlinfer.ray")

    def _infer_local_ray_custom_resources(device_type, world_size):
        if device_type == "ascend":
            n = None
            try:
                npu_mod = getattr(torch, "npu", None)
                if npu_mod is not None and callable(
                    getattr(npu_mod, "device_count", None)
                ):
                    n = int(npu_mod.device_count())
                    if n <= 0:
                        n = None
            except Exception:
                n = None
            if n is None:
                vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
                if vis:
                    n = len([x for x in vis.split(",") if x.strip() != ""])
            if n is None or n <= 0:
                n = int(world_size)
                logger.warning(
                    "Could not detect NPU count; registering Ray resource "
                    "NPU=%d from world_size.",
                    n,
                )
            return {"NPU": float(n)}

        if device_type == "camb":
            n = None
            try:
                mlu = getattr(torch, "mlu", None)
                if mlu is not None and callable(
                    getattr(mlu, "device_count", None)
                ):
                    n = int(mlu.device_count())
                    if n <= 0:
                        n = None
            except Exception:
                n = None
            if n is None or n <= 0:
                n = int(world_size)
                logger.warning(
                    "Could not detect MLU count; registering MLU=%d.", n
                )
            return {"MLU": float(n)}

        return None

    def _patched_init_ray_cluster(
        world_size, ray_address=None, dp=1, device_type="cuda"
    ):
        """Register custom resources at ray.init() for local clusters."""
        import ray

        if not ray.is_initialized():
            num_cpus = world_size
            object_store_memory = _ray_mod._get_obj_store_memory(dp=dp)
            init_kwargs = dict(
                ignore_reinit_error=True,
                num_cpus=num_cpus,
                object_store_memory=object_store_memory,
            )
            if ray_address is not None:
                init_kwargs["address"] = ray_address
            if ray_address is None:
                custom_res = _infer_local_ray_custom_resources(
                    device_type, world_size
                )
                if custom_res:
                    init_kwargs["resources"] = custom_res
            try:
                ray.init(**init_kwargs)
            except ValueError as e:
                if (
                    e.args is not None
                    and len(e.args) >= 1
                    and e.args[0]
                    == (
                        "When connecting to an existing cluster, num_cpus "
                        "and num_gpus must not be provided."
                    )
                ):
                    ray.init(address=ray_address, ignore_reinit_error=True)
                else:
                    raise

        device_str = _ray_mod.get_device_str(device_type)
        current_placement_group = ray.util.get_current_placement_group()
        owned_pg = False
        if not current_placement_group:
            num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
            if world_size > num_devices_in_cluster:
                _ray_mod.logger.warning(
                    "The number of required %ss exceeds the total "
                    "number of available %ss in the placement group.",
                    device_str,
                    device_str,
                )
            placement_group_specs = [{device_str: 1.0} for _ in range(world_size)]
            current_ip = ray.util.get_node_ip_address()
            placement_group_specs[0][f"node:{current_ip}"] = 0.001
            current_placement_group = ray.util.placement_group(
                placement_group_specs, strategy="PACK"
            )
            _ray_mod._wait_until_pg_ready(current_placement_group)
            owned_pg = True

        assert current_placement_group is not None
        placement_group = current_placement_group
        return placement_group, owned_pg

    _ray_mod.init_ray_cluster = _patched_init_ray_cluster
```

## 启用方式

在 `vendor_device_init()` 的 Ascend 分支中调用该 patch：

```python
def vendor_device_init():
    import_vendor_module(vendor_name)
    patch_compiled_func()
    patch_async_sampling_logits()
    if vendor_name in ["camb", "ascend"]:
        patch_contiguous_cache_engine()
    if vendor_name == "ascend":
        patch_rejection_sampler()
        patch_state_cache_engine()
        patch_gated_delta_net()
        patch_qwen3_5()
        patch_ray_init()
```

## 验证方法

启用 patch 后，启动推理任务前后可以检查 Ray 资源：

```bash
python - <<'PY'
import ray

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

print(ray.cluster_resources())
PY
```

期望输出中包含类似：

```text
{'CPU': 4.0, 'NPU': 4.0, ...}
```

也可以使用：

```bash
ray status
```

确认 `NPU` 资源数量是否符合预期。

## 注意事项

- 该 patch 主要针对 `ray_address is None` 的本地 Ray 初始化路径。
- 如果使用 `ray start` 预先启动集群，建议优先检查 `ray start` 时是否正确
  注册了自定义资源。
- 该 patch 不改变 NPU 可见设备绑定逻辑，只负责把 Ray 调度所需的 `NPU`
  资源数量注册进去。
- 如果 `torch.npu.device_count()` 和 `ASCEND_RT_VISIBLE_DEVICES` 都无法获取
  设备数量，patch 会退回使用 `world_size`。
- 在 Ascend A3 上目前没有复现该问题，因此不建议默认打开；建议只在出现
  上述 placement group 调度失败时尝试。
