# Copyright (c) 2024, DeepLink. All rights reserved.
# decorator usage
def register_ops(registry):
    def wrapped_func(ops_func):
        registry[ops_func.__name__] = ops_func
        return ops_func

    return wrapped_func
