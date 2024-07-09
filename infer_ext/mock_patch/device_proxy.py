from functools import partial
import torch
from infer_ext.vendor import device_type


class _MetaDeviceType(type):
    _torch_device = torch.device

    def __instancecheck__(cls, inst):
        if isinstance(inst, cls._torch_device):
            return True
        return False


# csrc/Device.cpp THPDevice_pynew:
# "Device(Device device)" Device type can be Device, Long, String
# "Device(c10::string_view type, int64_t? index=-1)"
class _DIPUDevice(metaclass=_MetaDeviceType):
    @staticmethod
    def __replacedipu(arg):
        if "cuda" in arg:
            arg = arg.replace("cuda", device_type)
        return arg

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], int):
            # modify default int device type only when "mock cuda".
            dev_name = device_type + ":" + str(args[0])
            _device = _MetaDeviceType._torch_device(dev_name)
            return _device
        # handle device as str
        if len(args) >= 1 and isinstance(args[0], str):
            argList = list(args)
            argList[0] = cls.__replacedipu(args[0])
            args = tuple(argList)
        # handle parameter type: str, not support int type but str and device
        deviceValue = kwargs.get("type", None)
        if deviceValue != None and isinstance(deviceValue, str):
            kwargs["type"] = cls.__replacedipu(deviceValue)
        _device = _MetaDeviceType._torch_device(*args, **kwargs)
        return _device


# always patch: device class is immutable, cannot directly patch __new__ method on python layer.
torch.device = _DIPUDevice

def GetDeviceProxy(rawfunc, pos=0, name="device", caller="obj"):
    def _replaceDevice(args, kwargs):
        # pos device
        if (
            pos >= 0
            and pos < len(args)
            and (isinstance(args[pos], int) or isinstance(args[pos], str))
        ):
            argList = list(args)
            argList[pos] = torch.device(args[pos])
            args = tuple(argList)
        deviceValue = kwargs.get(name, None)
        if deviceValue != None and (
            isinstance(deviceValue, int) or isinstance(deviceValue, str)
        ):
            kwargs[name] = torch.device(deviceValue)
        return args, kwargs

    def _deviceToStr(args, kwargs):
        # pos device
        if pos >= 0 and pos < len(args) and isinstance(args[pos], torch.device):
            argList = list(args)
            argList[pos] = args[pos].type
            args = tuple(argList)
        deviceValue = kwargs.get(name, None)
        if isinstance(deviceValue, torch.device):
            kwargs[name] = deviceValue.type
        return args, kwargs

    def _proxyFuncInst(self, *args, **kwargs):
        args, kwargs = _replaceDevice(args, kwargs)
        return rawfunc(self, *args, **kwargs)

    def _proxyFuncStatic(*args, **kwargs):
        args, kwargs = _replaceDevice(args, kwargs)
        return rawfunc(*args, **kwargs)

    # class __new__ always pass cls parameter to args
    def _proxyNewClass(cls, *args, **kwargs):
        args, kwargs = _replaceDevice(args, kwargs)
        return rawfunc(cls, *args, **kwargs)

    # return device in string
    def _proxyFuncStaticStr(self, *args, **kwargs):
        args, kwargs = _replaceDevice(args, kwargs)
        args, kwargs = _deviceToStr(args, kwargs)
        return rawfunc(self, *args, **kwargs)

    if caller == "static":
        return _proxyFuncStatic
    elif caller == "class_new":
        return _proxyNewClass
    elif caller == "str_static":
        return _proxyFuncStaticStr
    else:
        return _proxyFuncInst

GetDeviceStaticProxy = partial(GetDeviceProxy, pos=-1, name="device", caller="static")
