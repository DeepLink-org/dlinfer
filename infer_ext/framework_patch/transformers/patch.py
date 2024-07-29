import types

func_patches = {}
class_method_patches = {}

def register_func_patch(target_func):
    def decorator(replacement_func):
        func_patches[target_func] = replacement_func
        return replacement_func
    return decorator

def register_class_method_patch(target_class, target_method_name):
    def decorator(replacement_method):
        class_method_patches[(target_class, target_method_name)] = replacement_method
        return replacement_method
    return decorator

def apply_patches():
    for target_func, replacement_func in func_patches.items():
        module = target_func.__module__
        func_name = target_func.__name__
        setattr(__import__(module), func_name, replacement_func)
    for (target_class, target_method_name), replacement_method in class_method_patches.items():
        setattr(target_class, target_method_name, replacement_method)

import internlm2

apply_patches()
