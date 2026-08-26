# Ascend CUDAGraph 全局状态与 RL Re-capture 生命周期

本文说明 DLINFER Ascend CUDAGraph 实现中以下两个全局变量的历史用途、生命周期，以及它们与 RL rollout re-capture 的关系：

```python
_graph_params: Optional[GraphParams] = None
_graph_capture_sizes: set[int] = None
```

核心结论如下：

- `_graph_params` 历史上保存旧版 Attention graph-task 更新所需的图资源，必须在 rollout reset 时清理。
- `_graph_capture_sizes` 当前只有赋值和清空，没有任何读取位置，因此没有实际控制作用。
- RL re-capture 修复真正依赖的是
  `_get_capture_batch_size_impl.cache_clear()`，而不是
  `_graph_capture_sizes` 变量本身。
- “capture size 需要跨 rollout 保存”指 capture-size 策略或配置，不是旧图的 handle、event 和 Tensor 引用。

> 实现状态（2026-08-17）：本文第 6 节的重构已经落地。旧 ATB
> graph-task 路径以及 `_graph_params`、`_graph_capture_sizes` 已删除；
> capture-size 计算已成为纯函数，显式配置由 `CacheConfig` 跨 rollout 保存。

## 1. `_graph_params` 的历史用途

`_graph_params` 最早在提交
[`5c474737`](https://github.com/DeepLink-org/dlinfer/commit/5c4747371b6bd3f39a71656c74e99e3429f259d1)
中引入，用于支持旧版 ATB Attention 的 graph-task 动态参数更新：

```python
@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]
    is_mla: bool
```

其中的字典以 capture size 为 key。例如捕获 batch size 8 的图时，对应资源存放在：

```python
events[8]
handles[8]
attn_params[8]
```

Attention capture 期间，每一层会注册以下信息：

- Attention 输入 Tensor；
- KV cache Tensor；
- `block_table`；
- `kv_seq_len`；
- 输出 Tensor；
- graph-task handle；
- 同步 event。

旧版 replay 时，Graph Runner 根据当前 graph size 找到这些对象：

```python
graph_params.attn_params[runtime_size]
graph_params.handles[runtime_size]
graph_params.events[runtime_size]
```

随后通过低层 graph-task API 更新 Attention 的运行时参数：

```python
torch.npu.graph_task_update_begin(...)
torch.ops.atb._npu_paged_attention(...)
torch.npu.graph_task_update_end(...)
```

因此，`_graph_params` 本质上是旧 ATB Attention kernel 与 Graph
Runner 之间的全局 side channel。它不是普通配置元信息，而是持有真实图资源和
Tensor 引用的对象。

## 2. 为什么 rollout 之间必须清理 `_graph_params`

RL rollout 的典型生命周期如下：

```text
rollout N 推理
  -> sleep / 更新权重
  -> reset graph
  -> wakeup
  -> rollout N+1 重新 capture
```

模型 sleep 或更新权重后：

- 原权重 Tensor 地址可能发生变化；
- KV cache 可能被释放或重新分配；
- captured graph 中保存的输入输出地址可能已经失效；
- ATB task handle 和 event 属于旧图；
- `attn_params` 中保存的 Tensor 引用会阻止显存释放。

所以 `_graph_params` 不能跨 rollout 复用。

提交
[`d0f60279`](https://github.com/DeepLink-org/dlinfer/commit/d0f60279a684de7dd37f0cab59d5065747faf598)
为此增加了 `clear_graph_params()`：

```python
attn_params.clear()
handles.clear()
events.clear()
workspaces.clear()
_graph_params = None
```

如果错误地保留 `_graph_params`，可能导致：

- 显存泄漏；
- 使用旧 KV cache 地址；
- 使用已失效的权重地址；
- graph replay 卡住；
- 权重更新后仍得到旧结果。

从生命周期上可以把状态分成两类：

```text
应该跨 rollout 保存：
    capture-size 策略、max_batches、模型类型

不应该跨 rollout 保存：
    NPUGraph、task handle、event、Tensor 引用、output buffer
```

## 3. RL re-capture 问题的根因

历史实现把 capture-size 计算和 `_graph_params` 初始化放进了同一个带缓存函数：

```python
@functools.lru_cache
def _get_capture_batch_size_impl(max_batches):
    ...
    set_graph_params(set(ret))
    return ret
```

这里混合了两种不同职责：

1. 计算 capture sizes；
2. 初始化 `_graph_params`。

第一次 rollout 使用 `max_batches=256` 时：

```text
_get_capture_batch_size_impl(256)
  -> 函数体执行
  -> set_graph_params(...)
  -> lru_cache 保存返回值
```

sleep/reset 时：

```text
clear_graph_params()
  -> _graph_params = None
```

第二次 rollout 仍使用 `max_batches=256` 时：

```text
_get_capture_batch_size_impl(256)
  -> 命中 lru_cache
  -> 直接返回 capture-size 列表
  -> 函数体不执行
  -> set_graph_params() 没有再次调用
  -> _graph_params 仍是 None
```

接下来 Attention capture 如果执行：

```python
get_graph_params().events[...]
```

就会访问 `None`。这才是 re-capture 问题的根因。

## 4. `_graph_capture_sizes` 的实际作用

`_graph_capture_sizes` 由
[`PR #319: fix re-capture in RL`](https://github.com/DeepLink-org/dlinfer/pull/319)
引入：

```python
_graph_capture_sizes: set[int] = None
```

初始化时进行赋值：

```python
_graph_capture_sizes = aclgraph_capture_sizes
```

清理时执行：

```python
_graph_capture_sizes = None
_get_capture_batch_size_impl.cache_clear()
```

检查该 PR 的提交快照、当前开发分支和 `origin/main` 后可以确认，`_graph_capture_sizes` 只有以下操作：

- 声明；
- `global` 引用；
- 赋值；
- 清空。

代码中没有读取它的 getter，也没有用它重新初始化 `GraphParams`。因此，按照当前实现，
`_graph_capture_sizes` 是一个冗余的 bookkeeping 变量，对 re-capture
没有功能性贡献。

PR #319 真正修复问题的是：

```python
_get_capture_batch_size_impl.cache_clear()
```

它保证下一次 rollout 即使使用相同的 `max_batches`，函数体也会重新执行，并再次调用 `set_graph_params()`。

该 PR 没有正文和讨论，只有 8 行修改。因此无法从历史材料证明作者计划在其他地方读取 `_graph_capture_sizes`；从现有代码只能确认它没有参与任何控制逻辑。

## 5. 如何理解“capture size 要跨 rollout 保存”

这里的 capture size 应当理解为 engine 需要捕获哪些 batch size，例如：

```text
[1, 2, 4, 8, 16, 32]
```

这是配置或 shape policy，应该跨 rollout 保留，因为它决定 wakeup 后需要重新捕获哪些图。

但它不等同于 `_graph_capture_sizes` 这个全局变量。更合理的持久化来源是：

```python
CacheConfig.max_batches
CacheConfig.cudagraph_capture_batch_sizes
```

LMDeploy 后来在提交
[`4f25485e`](https://github.com/InternLM/lmdeploy/commit/4f25485e218d5e0938240da1d4fccbb426573a89)
中正式把 capture sizes 放进了 `CacheConfig`：

```python
cudagraph_capture_batch_sizes: list[int] | None
```

合理的状态归属如下：

```text
CacheConfig                         跨 rollout 保留
    `-- cudagraph_capture_batch_sizes

AscendGraphRunner                   rollout 之间 reset
    `-- _runner_map
          `-- AscendSingleGraphRunner
                `-- NPUGraph

旧版全局 _graph_params             rollout 之间销毁
    |-- handles
    |-- events
    `-- Tensor refs
```

重构后的 Ascend DLINFER `get_capture_batch_sizes()` 会优先读取 LMDeploy 的
`CacheConfig.cudagraph_capture_batch_sizes`；未显式配置时，才调用 Ascend 的
`_get_capture_batch_size_impl()` 生成默认尺寸。

## 6. 本次重构结果

本次重构没有只做以下机械删除：

```text
删除 _graph_params
删除 _graph_capture_sizes
删除 cache_clear()
```

同时完成了以下生命周期迁移：

1. capture-size 计算已经成为纯函数：

   ```python
   @functools.lru_cache
   def _get_capture_batch_size_impl(max_batches):
       ...
       return ret
   ```

   其中不能再调用 `set_graph_params()`。

2. capture-size 策略保存在 `CacheConfig` 中，可以跨 rollout 保留。
3. `op_backend.py` 根据 K/V head dim 显式判断并向 Graph Runner 传递 `is_mla`。
4. Dense decode 固定使用 FIA，MLA decode 和 paged prefill 固定使用 FIA v2。
5. Graph replay 固定使用 `NPUGraph.update()`；旧 ATB graph-task Attention 路径已删除。
6. `_graph_params` 和从未被读取的 `_graph_capture_sizes` 已整体删除。
7. `AscendGraphRunner.reset()` 先调用基类 reset，清空 rollout 局部的
   `padding_batch_size`，但不清除 `CacheConfig` 中的 capture sizes。
8. 增加 capture-size 纯函数、reset 配置保留、FIA v2 graph replay 和版本门槛测试。

完整 RL 场景仍建议执行以下端到端回归：

```text
capture -> inference
reset/sleep
wakeup
使用相同 max_batches 再次 capture
inference
重复两到三轮
```

测试需要验证：

- 不出现 `_graph_params is None`；
- 不使用旧权重；
- 不持有旧 KV cache；
- capture sizes 与第一次一致；
- graph replay 精度正确；
- 多轮 sleep/wakeup 后没有显存持续增长。

## 7. 最终结论

最终可以删除 `_graph_params` 和 `_graph_capture_sizes`，但必须保留它们背后的两类语义：

- `_graph_params` 背后的旧图资源生命周期：新路径中由每个 `NPUGraph` 自己管理，并在 reset 时释放。
- `_graph_capture_sizes` 名字所暗示的 capture-size 策略：迁移到持久的
  `CacheConfig`，不能随着 rollout 丢失。

换句话说，需要保留的是生命周期语义和配置来源，而不是这两个模块级全局变量本身。
