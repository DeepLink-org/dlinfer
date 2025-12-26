# 华为昇腾 LMDeploy 多节点部署指南（Ray + PyTorchEngine）

本文介绍如何在 **Atlas 800T A2（2 节点 × 8 卡）** 环境下，通过 **Ray** 组织多机资源，
并使用 **LMDeploy PyTorchEngine** 以 **tp=16** 的方式启动推理服务或调用 `pipeline`。

## 0. 适用范围与前置条件

- **适用范围**：多机多卡（跨节点）Tensor Parallel（TP）推理。
- **环境一致性**：建议各节点的驱动/CANN/容器镜像版本一致或兼容。
- **网络要求**：节点间网络互通、丢包低、时延稳定；TP 对网络质量非常敏感。
- **重要限制**：当前多机仅支持 **`eager` 模式**（见第 4 节启动参数）。

## 1. 创建 Docker 容器（可选）

为保证各节点运行环境一致，建议使用 Docker。以下命令需在**每个节点**执行；请按需补充模型目录挂载（例如 `-v /path/to/model:/models`）。

```bash
docker run -it \
  --net=host \
  --entrypoint /bin/bash \
  --shm-size=500g \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --privileged=true \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons \
  -v /usr/local/sbin/:/usr/local/sbin \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
  -v /var/log/npu/slog/:/var/log/npu/slog/ \
  -v /var/log/npu/profiling/:/var/log/npu/profiling/ \
  -v /var/log/npu/dump/:/var/log/npu/dump/ \
  -v /var/log/npu/:/usr/slog \
  -v /lib/modules:/lib/modules \
  crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest
```

## 2. 使用 Ray 组建多节点集群

### 2.1 环境变量说明（建议显式配置）

- **`GLOO_SOCKET_IFNAME`**：PyTorch/Gloo 通信使用的网卡名（例如 `eth0`）。
- **`TP_SOCKET_IFNAME`**：TP 通信使用的网卡名（通常与 `GLOO_SOCKET_IFNAME` 一致）。
- **`HCCL_IF_IP`**：HCCL 选择的网卡 **IP**（注意是 IP，不是网卡名）。
- **`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`**：避免 Ray 自动改写可见设备集合
  （Ascend 场景下建议保留）。

> [!TIP]
> 不确定网卡名/IP 时，可在宿主机或容器内用 `ip a` 查看；务必确保多机之间该网段互通。

### 2.2 启动 Head 节点

选择其中一个节点作为 Head 节点，并在该节点的容器中执行（将 `PORT`、`IFNAME`、`NODE_IP` 替换为实际值）：

```bash
# Head node
export GLOO_SOCKET_IFNAME=IFNAME
export TP_SOCKET_IFNAME=IFNAME
export HCCL_IF_IP=NODE_IP
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

ray start --head --port=PORT
```

### 2.3 Worker 节点加入集群

在其他节点容器中执行（将 `HEAD_NODE_IP`、`PORT`、`NODE_IP`、`IFNAME` 替换为实际值）：

```bash
# Worker node
export GLOO_SOCKET_IFNAME=IFNAME
export TP_SOCKET_IFNAME=IFNAME
export HCCL_IF_IP=NODE_IP
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

ray start --address=HEAD_NODE_IP:PORT --node-ip-address=NODE_IP
```

### 2.4 集群状态检查与清理（可选）

在 Head 节点可通过以下命令确认 worker 是否已加入：

```bash
ray status
```

如需清理（Head/Worker 都需要各自执行）：

```bash
ray stop
```

## 3. 配置 Rank Table（`ASCEND_RANK_TABLE_FILE_PATH`）

LMDeploy Ascend 多机需要配置 **`ASCEND_RANK_TABLE_FILE_PATH`**，指向本机可访问的 rank table 文件路径。
rank table 的字段含义和配置规范可参考：
[rank table 配置资源信息](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/hccl/hcclug/hcclug_000067.html)。

以下为双机示例（为便于阅读，示例仅列出每机 2 卡，实际请补齐 8 卡并确保 `rank_id` 全局唯一）：

```json
{
  "status": "completed",
  "version": "1.0",
  "server_count": "2",
  "server_list": [
    {
      "server_id": "node_0",
      "device": [
        {
          "device_id": "0",
          "device_ip": "192.168.1.8",
          "device_port": "16667",
          "rank_id": "0"
        },
        {
          "device_id": "1",
          "device_ip": "192.168.1.9",
          "device_port": "16667",
          "rank_id": "1"
        }
      ]
    },
    {
      "server_id": "node_1",
      "device": [
        {
          "device_id": "0",
          "device_ip": "192.168.2.8",
          "device_port": "16667",
          "rank_id": "2"
        },
        {
          "device_id": "1",
          "device_ip": "192.168.2.9",
          "device_port": "16667",
          "rank_id": "3"
        }
      ]
    }
  ]
}
```

> [!WARNING]
> 后续我们将提供自动生成 `rank table` 文件的脚本。
> [!IMPORTANT]
> 多机场景下建议将 rank table 文件放在**所有节点相同的路径**，
> 并在启动服务前确保相关进程环境中已设置 `ASCEND_RANK_TABLE_FILE_PATH`
> （例如在启动 `ray start`/`lmdeploy` 前统一 `export`）。

## 4. 启动与调用（LMDeploy）

### 4.1 启动 API 服务（在 Head 节点执行）

```bash
export ASCEND_RANK_TABLE_FILE_PATH=/path/to/rank_table.json

lmdeploy serve api_server \
  $CONTAINER_MODEL_PATH \
  --backend pytorch \
  --device ascend \
  --eager-mode \
  --tp 16
```

### 4.2 使用 `pipeline` 接口

```python
import os
from lmdeploy import pipeline, PytorchEngineConfig

os.environ["ASCEND_RANK_TABLE_FILE_PATH"] = "/path/to/rank_table.json"

if __name__ == "__main__":
  model_path = "/path/to/model"
  backend_config = PytorchEngineConfig(tp=16, device_type="ascend", eager_mode=True)
  with pipeline(model_path, backend_config=backend_config) as pipe:
    outputs = pipe("Hakuna Matata")
    print(outputs)
```

> [!IMPORTANT]
> 当前多机仅支持 **`eager` 模式**，请务必设置 `--eager-mode` / `eager_mode=True`。

## 5. 多机通信与 HCCL 排障（强烈建议部署前检查）

为获得更好的稳定性与性能，建议配置更好的网络环境（例如 [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand)）。

若出现 “HCCL 集群通信失败” 等问题，可按以下步骤快速定位（参考 Ascend 案例库：
[HCCL集群通信失败](https://www.hiascend.com/document/caselibrary/detail/topic_0000001953463657)）：

### 5.1 检查 NPU device IP 是否互通

以双机（A、B），每机 8 卡为例：

1. 在节点 A 查询各卡 device IP：

   ```bash
   for i in {0..7}; do hccn_tool -i $i -ip -g; done
   ```

1. 在节点 B 使用本机某张卡去 ping 节点 A 的 device IP（示例用 B 节点的 0 卡）：

   ```bash
   hccn_tool -i 0 -ping -g address 192.x.x.x
   ```

若回显包含 “0.00% packet loss” 则说明可达；否则需要检查网络配置（路由/掩码/VLAN/防火墙等）。

> [!TIP]
> 若 device IP 为 IPv6，可使用 `-inet6` 参数：
>
> - 查询：`for i in {0..7}; do hccn_tool -i $i -ip -inet6 -g; done`
> - ping：`hccn_tool -i 0 -ping -inet6 -g ipv6_address x:x:x:x`

### 5.2 检查各节点 TLS 配置是否一致

在两台节点分别执行，确认输出一致：

```bash
for i in {0..7}; do hccn_tool -i $i -tls -g | grep switch; done
```

若 TLS 配置不一致，需要统一配置后再重试（更详细的处理建议以案例库说明为准：见
[HCCL集群通信失败](https://www.hiascend.com/document/caselibrary/detail/topic_0000001953463657)）。
