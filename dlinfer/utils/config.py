# Copyright (c) 2024, DeepLink. All rights reserved.
class Config:
    def __init__(self, **kwargs):
        self._config = kwargs

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_config":
            super().__setattr__(name, value)
        else:
            self._config[name] = value

    def __repr__(self):
        return repr(self._config)
