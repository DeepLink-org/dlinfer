from abc import ABCMeta, abstractmethod

from dlinfer.graph.dicp.dynamo_bridge.torch_version import is_torch_251_or_higher

if is_torch_251_or_higher:
    from torch._inductor.async_compile import AsyncCompile
else:
    from torch._inductor.codecache import AsyncCompile


class DeviceCompileJob:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_key():
        pass

    @abstractmethod
    def get_compile_result():
        pass


class DeviceKernelCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def get_kernel(cls, device_compile_job):
        key = device_compile_job.get_key()
        if key not in cls.cache:
            loaded = device_compile_job.get_compile_result()
            cls.cache[key] = loaded
            cls.cache[key].key = key
        return cls.cache[key]


class AsyncCompileKernel(AsyncCompile):
    def compile_kernel(self, device_compile_job):
        return DeviceKernelCache.get_kernel(device_compile_job).run
