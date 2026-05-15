# 华为昇腾 LMDeploy 多节点部署指南

LMDeploy + DLInfer 在昇腾上的多节点部署分两类：**TP 在节点内**(常规)和 **TP 跨节点**(老版本兼容)。本文档分别给出两类的启动流程与脚本。

## 0. 选择部署模式

每个 DP 实际占用的卡数(也就是每个 DP 内部的 TP size) = 服务总卡数 ÷ `--dp`。

> [!NOTE]
> 关于 lmdeploy 的 `--tp` 与 `--dp` / `--ep` 的关系：
>
> `--dp` 默认值是 `1`，`--ep` 默认值是 `1`。
>
> - `--ep` 不设(默认为 `1`)时：**总卡数 = `--tp`**
> - `--ep` 设了(`> 1`)时：**总卡数 = `--tp × --dp`**
>
> 每个 DP 内部的 TP size = 总卡数 ÷ `--dp`。
>
> 例：`--tp 4 --dp 2` 表示服务用 4 张卡，2 个 DP，每个 DP 的 TP size 是 2。
>
> 例：`--tp 16 --dp 2 --ep 32` 表示服务用 32 张卡，2 个 DP，每个 DP 的 TP size 是 16。

判断走哪种模式：

- **每个 DP 内部的 TP size ≤ 单节点 NPU 数**(即每个 DP 都装得进一个节点)→ 走 [§2 TP 在节点内](#2-tp-在节点内常规)
- **每个 DP 内部的 TP size > 单节点 NPU 数**(单个 DP 就需要跨节点)→ 走 [§3 TP 跨节点](#3-tp-跨节点)

两种模式都依赖 [§1 通用前置](#1-通用前置两种模式都需要)。

---

## 1. 通用前置(两种模式都需要)

### 1.1 容器镜像与 docker run

每个节点起一个相同镜像的容器，配置一致是必须的。

```bash
docker run -it \
  --net=host \
  --shm-size=500g \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  ... # 列齐你要用的所有 NPU 设备文件
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
  <image>
```

### 1.2 host 侧 NPU 网络配置

NPU 网卡的 device IP / netmask / gateway 由 host 维护者通过 `/etc/hccn.conf` 配置，host 启动时下发到设备。本文档假设这一步已经做好。

容器里可以用 `hccn_tool` 验证(此工具仅作诊断用，推理运行时不依赖)：

```bash
# 列出本机每张 NPU 的 device_ip
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done    # 单机 8 卡
for i in {0..15}; do hccn_tool -i $i -ip -g | grep ipaddr; done   # 单机 16 卡
```

### 1.3 跨节点 device IP 互通检查

部署前**强烈建议**做一次。两机各取一张卡互 ping(走 NPU 网卡，非 host NIC)：

```bash
# 节点 A：查询本机 0 号卡 device_ip
hccn_tool -i 0 -ip -g
# 假设返回 ipaddr:192.168.1.8

# 节点 B：用本机 0 号卡 ping 节点 A 的 0 号卡
hccn_tool -i 0 -ping -g address 192.168.1.8
```

回显 `0.00% packet loss` 才说明 NPU 网络层互通。否则要先排查
host 侧路由 / 掩码 / VLAN / 防火墙——这一步不通后续 HCCL 一定挂。

IPv6 用 `-inet6` 参数。

### 1.4 关键环境变量速查表

每个 server 启动脚本里都需要 export 的：

| 变量 | 作用 | 常见取值 |
| --- | --- | --- |
| `HCCL_IF_IP` | HCCL 跨机引导用的 host IP(本节点) | 本机 host IP |
| `GLOO_SOCKET_IFNAME` | PyTorch Gloo 通信走的网卡名 | `eth0` |
| `TP_SOCKET_IFNAME` | TP rendezvous 走的网卡名 | 通常同上 |
| `HCCL_SOCKET_IFNAME` | HCCL TCP 引导走的网卡名 | 通常同上 |
| `HCCL_BUFFSIZE` | HCCL 通信缓冲区大小 (MB) | `1024`，跨机 TP 可调到 `2048` |
| `HCCL_CONNECT_TIMEOUT` | HCCL 跨机连接超时(秒) | `7200` |
| `HCCL_OP_EXPANSION_MODE` | HCCL 算子优化策略 | `AIV` |
| `PYTORCH_NPU_ALLOC_CONF` | 内存分配策略 | `expandable_segments:True` |
| `LMDEPLOY_DP_MASTER_ADDR` | DP rendezvous 的 master 节点 IP | master 节点 host IP |
| `LMDEPLOY_DP_MASTER_PORT` | DP rendezvous 端口 | `29555` 等任选 |
| `ASCEND_RT_VISIBLE_DEVICES` | **可选**，限定本进程可见的 NPU | 如 `0,1,2,3,4,5,6,7` |

---

## 2. TP 在节点内(常规)

### 2.1 适用拓扑示例

| 总卡数 | 配置 | 拓扑 |
| --- | --- | --- |
| 32 | `--tp 16 --dp 2 --ep 32` | 2 节点 × 16 NPU，每个 DP 占满一台 |
| 32 | `--tp 8 --dp 4 --ep 32` | 2 节点 × 16 NPU，每节点跑 2 个 DP |
| 16 | `--tp 8 --dp 2 --ep 16` | 2 节点 × 8 NPU，每个 DP 占满一台 |
| 16 | `--tp 4 --dp 4 --ep 16` | 2 节点 × 8 NPU，每节点跑 2 个 DP |

通用约束：**每个 DP 内部的 TP size ≤ 单节点 NPU 数**。

### 2.2 启动步骤

3 步：proxy + 每节点 server。**每节点的脚本要用本节点对应的 IP 和 `--node-rank`**。

#### Step 1：在 master 节点起 proxy

```bash
lmdeploy serve proxy \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --routing-strategy 'min_expected_latency' \
    --serving-strategy Hybrid \
    --log-level INFO
```

#### Step 2：在 node 0 起 server

```bash
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
export RAY_DEDUP_LOGS=0

# 网络 env(HCCL 自己用)
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export HCCL_IF_IP=10.0.0.1                # ← node 0 的 host IP

# 可选：限定本节点可见 NPU
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# DP rendezvous：master 节点 host IP，两节点都填同一值；node-rank 是本节点的全局编号
LMDEPLOY_DP_MASTER_ADDR=10.0.0.1 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    /path/to/model \
    --backend pytorch \
    --device ascend \
    --tp 16 --dp 2 --ep 32 \
    --nnodes 2 --node-rank 0 \
    --dtype bfloat16 \
    --session-len 65535 \
    --log-level WARNING \
    --proxy-url http://10.0.0.1:23333
```

#### Step 3：在 node 1 起 server

```bash
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
export RAY_DEDUP_LOGS=0

export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export HCCL_IF_IP=10.0.0.2                # ← node 1 的 host IP

# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# DP rendezvous 仍指向 master 节点（与 node 0 一致）
LMDEPLOY_DP_MASTER_ADDR=10.0.0.1 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    /path/to/model \
    --backend pytorch \
    --device ascend \
    --tp 16 --dp 2 --ep 32 \
    --nnodes 2 --node-rank 1 \
    --dtype bfloat16 \
    --session-len 65535 \
    --log-level WARNING \
    --proxy-url http://10.0.0.1:23333
```

服务起来之后，客户端请求发到 `http://<master 节点 IP>:23333`(proxy 的端口)。

---

## 3. TP 跨节点

这个模式是为了兼容老版本部署方式而保留的，有时也可以用它来快速验证多节点推理的可行性。
**实际生产推理中大概率用不到**——如果你的 TP size 能塞进一个节点，
直接走 [§2](#2-tp-在节点内常规) 即可。

### 3.1 与 [§2](#2-tp-在节点内常规) 的关键差异

跨节点 TP 一般用作纯 TP 部署，此时 `--dp` 通常是 `1`；而 TP 在节点内的部署里，`--dp` 取决于实际拓扑，并不固定。

| 项 | TP 在节点内 | TP 跨节点 |
| --- | --- | --- |
| 是否需要预先起跨节点 Ray cluster | 否 | 是 |
| 每节点是否需要起 lmdeploy serve | 是 | 否(只在 master 节点起一次) |
| `--node-rank` / `--nnodes` | 需要 | 不需要 |
| 典型 `--dp` 值 | 取决于实际拓扑 | `1`(纯 TP) |
| proxy | `dp > 1` 时需要 | 不需要 |

### 3.2 启动步骤

4 步：node 0 起 ray head，node 1 加入 ray cluster，确认集群，master 节点起 lmdeploy。

#### Step 1：在 node 0 起 ray head

```bash
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export HCCL_IF_IP=10.0.0.1                  # ← node 0 的 host IP
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

ray start --head --port=22345 --node-ip-address=10.0.0.1
```

#### Step 2：在 node 1 加入 cluster

```bash
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export HCCL_IF_IP=10.0.0.2                  # ← node 1 的 host IP
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

ray start --address=10.0.0.1:22345 --node-ip-address=10.0.0.2
```

#### Step 3：确认集群状态

在 master 节点执行：

```bash
ray status
```

期望看到 2 个 node，资源里 NPU 总数 = 两节点 NPU 之和。

#### Step 4：在 master 节点起 lmdeploy

只在 master 节点跑(node 1 不需要跑任何 lmdeploy 命令，只贡献 Ray actor)：

```bash
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=2048                   # 跨机 TP 建议调大
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
export RAY_DEDUP_LOGS=0

export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export HCCL_IF_IP=10.0.0.1

lmdeploy serve api_server \
    /path/to/model \
    --backend pytorch \
    --device ascend \
    --tp 16 --dp 1 \
    --dtype bfloat16 \
    --session-len 65535 \
    --log-level WARNING \
    --server-name 0.0.0.0 \
    --server-port 13333
```

`--tp` 是服务总卡数(此处 = 2 节点 × 8 NPU)，`--dp 1` 表示纯 TP。

服务起来之后，客户端请求发到 `http://<master 节点 IP>:13333`。

### 3.3 收尾

跑完想停掉，两节点都执行：

```bash
ray stop
```

---

## 4. 多机通信与 HCCL 排障(强烈建议部署前检查)

为获得更好的稳定性与性能，建议配置更好的网络环境(例如 [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand))。

若出现 "HCCL 集群通信失败" 等问题，可按以下步骤快速定位(参考 Ascend 案例库：[HCCL 集群通信失败](https://www.hiascend.com/document/caselibrary/detail/topic_0000001953463657))：

### 4.1 检查 NPU device IP 是否互通

以双机(A、B)，每机 8 卡为例：

1. 在节点 A 查询各卡 device IP：

   ```bash
   for i in {0..7}; do hccn_tool -i $i -ip -g; done
   ```

2. 在节点 B 使用本机某张卡去 ping 节点 A 的 device IP(示例用 B 节点的 0 卡)：

   ```bash
   hccn_tool -i 0 -ping -g address 192.x.x.x
   ```

若回显包含 "0.00% packet loss" 则说明可达；否则需要检查网络配置(路由 / 掩码 / VLAN / 防火墙等)。

> [!TIP]
> 若 device IP 为 IPv6，可使用 `-inet6` 参数：
>
> - 查询：`for i in {0..7}; do hccn_tool -i $i -ip -inet6 -g; done`
> - ping：`hccn_tool -i 0 -ping -inet6 -g ipv6_address x:x:x:x`

### 4.2 检查各节点 TLS 配置是否一致

在两台节点分别执行，确认输出一致：

```bash
for i in {0..7}; do hccn_tool -i $i -tls -g | grep switch; done
```

若 TLS 配置不一致，需要统一配置后再重试(更详细的处理建议以案例库说明为准：见 [HCCL 集群通信失败](https://www.hiascend.com/document/caselibrary/detail/topic_0000001953463657))。
