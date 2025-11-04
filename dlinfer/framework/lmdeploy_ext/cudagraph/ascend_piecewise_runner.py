"""
Piecewise Graph Runner
与lmdeploy的warmup机制集成
"""
import torch
from typing import List
from lmdeploy.pytorch.backends.graph_runner import GraphRunner
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

class AscendPiecewiseGraphRunner(GraphRunner):
    """
    基于torch.compile自定义backend的piecewise runner
    
    与lmdeploy集成：
    1. get_capture_batch_sizes(): 告知需要warmup的batch sizes
    2. __call__(): 被model_agent调用，首次触发capture，后续replay
    
    注意：
    - warmup由lmdeploy的model_agent统一管理
    - 参考：lmdeploy/pytorch/engine/model_agent.py:420-435
    """
    
    def __init__(self, model, model_config, cache_config, backend_config, device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        
        self.enable_graph = self._check_enable()
        self.compiled_model = None
        
        if self.enable_graph:
            self._setup()
        else:
            logger.info("Piecewise graph mode disabled (eager mode enabled)")
    
    def _setup(self):
        """设置piecewise模式"""
        logger.info("=" * 70)
        logger.info("Setting up Ascend Piecewise Graph Mode (vLLM-style)")
        logger.info("=" * 70)
        
        try:
            # 新方案：启用 enable_graph_mode，使用已注册的 dlinfer custom ops
            # 这样 dynamo 可以追踪完整图，不会因为未注册的 ops 报错
            logger.info("Step 1: Enabling dlinfer graph mode...")
            import dlinfer.graph
            dlinfer.graph.config.enable_graph_mode = True
            logger.info("✓ Graph mode enabled")
            logger.info("   → dlinfer custom ops (torch.ops.dlinfer::xxx) are now active")
            logger.info("   → dynamo can trace full graph")
            
            # Step 2: 编译 model（使用自定义 backend）
            logger.info("Step 2: Compiling model with DlinferPiecewiseBackend...")
            from dlinfer.graph.compilation.piecewise_backend import create_backend
            
            backend = create_backend()
            
            # 增加torch.compile的cache size限制
            # 因为我们需要为每个batch size编译一次
            import torch._dynamo
            original_cache_size = torch._dynamo.config.cache_size_limit
            torch._dynamo.config.cache_size_limit = 64  # 支持更多batch sizes
            logger.info(f"Increased torch._dynamo cache_size_limit: {original_cache_size} → 64")
            
            # 关键：现在可以使用 fullgraph=True
            # 因为 dlinfer custom ops 已注册（通过 register_custom_op）
            # dynamo 可以追踪到完整图，不会报参数匹配错误
            self.compiled_model = torch.compile(
                self.model,
                backend=backend,
                fullgraph=True,  # 可以追踪完整图（custom ops 已注册）
                dynamic=False,  # 每个batch size单独编译一次
            )
            
            logger.info("✓ Model compiled successfully")
            logger.info("=" * 70)
            logger.info("Setup complete. Waiting for warmup from model_agent...")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Failed to setup piecewise graph mode: {e}", exc_info=True)
            # 回退到 eager 模式
            logger.warning("Falling back to eager mode")
            self.enable_graph = False
            # 恢复 graph mode 设置
            import dlinfer.graph
            dlinfer.graph.config.enable_graph_mode = False
            self.compiled_model = self.model
    
    def get_capture_batch_sizes(self) -> List[int]:
        """
        返回需要warmup的batch sizes
        
        lmdeploy的model_agent会调用此方法获取warmup列表
        参考：model_agent.py:421
        
        策略：
        - 使用dynamic=False，每个batch size触发一次torch.compile
        - torch.compile会为每个batch size创建独立的compiled graph
        - ACL Graph会为每个batch size的每个子图capture一次
        
        返回较少的batch sizes以加速warmup
        """
        if not self.enable_graph:
            return []
        
        max_batch = self.cache_config.max_batches
        sizes = []
        
        # 简化策略：只warmup几个关键的batch sizes
        # Power of 2: 1, 2, 4, 8, 16
        bs = 1
        while bs <= min(16, max_batch):
            sizes.append(bs)
            bs *= 2
        
        # 如果max_batch较大，再添加几个大的
        if max_batch > 16:
            # 添加 32, 64, 128, 256
            for bs in [32, 64, 128, 256]:
                if bs <= max_batch and bs not in sizes:
                    sizes.append(bs)
        
        # 确保max_batch在列表中
        if max_batch not in sizes:
            sizes.append(max_batch)
        
        logger.info(f"Piecewise graph capture batch sizes: {sizes}")
        logger.info(f"Note: Each batch size will trigger one torch.compile (dynamic=False)")
        return sizes
    
    def __call__(self, **kwargs):
        """
        执行forward
        
        流程：
        - Prefill阶段（is_decoding=False）：直接使用eager模式（shape变化大）
        - Decode阶段（is_decoding=True）：使用piecewise graph模式（shape固定）
        
        注意：
        - model_agent的warmup会调用此方法触发capture
        - 推理阶段也通过此方法执行
        """
        if not self.enable_graph:
            # Eager模式
            return self.model(**kwargs)
        
        # 检查是否是decode阶段
        # 只有decode阶段才使用graph，prefill阶段走eager
        attn_metadata = kwargs.get('attn_metadata', None)
        is_decoding = getattr(attn_metadata, 'is_decoding', False) if attn_metadata else False
        
        if not is_decoding:
            # Prefill阶段：不使用graph，直接走eager模式
            logger.debug("Prefill phase: using eager mode (skipping graph)")
            return self.model(**kwargs)
        
        try:
            # Decode阶段：使用piecewise graph模式
            logger.debug("Decode phase: using piecewise graph mode")
            return self.compiled_model(**kwargs)
        except Exception as e:
            logger.error(f"Error in piecewise graph execution: {e}", exc_info=True)
            
            # 尝试回退到eager模式
            logger.warning("Attempting to fall back to eager mode")
            return self.model(**kwargs)
    
    def _check_enable(self) -> bool:
        """检查是否启用piecewise graph"""
        if self.backend_config.eager_mode:
            logger.info("Piecewise graph disabled: eager_mode=True")
            return False
        
        # 检查设备类型
        # 注意：在 Ascend 环境下，torch_npu 的兼容层可能导致 device.type 显示为 'cuda'
        # 所以这里只要不是明确的不支持设备就允许
        # 实际的 NPU 检查会在运行时通过 torch_npu.npu.is_available() 完成
        device_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        logger.debug(f"Device type: {device_type}")
        
        # 放宽检查：只要有 torch_npu 就认为是 NPU 环境
        try:
            import torch_npu
            if not torch_npu.npu.is_available():
                logger.warning("NPU not available, piecewise graph disabled")
                return False
        except ImportError:
            logger.warning("torch_npu not found, piecewise graph disabled")
            return False
        
        return True
    
    def __del__(self):
        """清理资源"""
        # 新方案不需要 unpatch（没有 patching 了）
        # 但需要恢复 enable_graph_mode 设置
        try:
            if hasattr(self, 'enable_graph') and self.enable_graph:
                import dlinfer.graph
                dlinfer.graph.config.enable_graph_mode = False
        except Exception:
            pass

