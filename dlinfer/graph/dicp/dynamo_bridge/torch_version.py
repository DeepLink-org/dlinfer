import torch
from packaging import version

torch_version = version.parse(torch.__version__).base_version

is_torch_200 = False
is_torch_210 = False
is_torch_220 = False
is_torch_231 = False
is_torch_251 = False

if torch_version.startswith("2.0"):
    is_torch_200 = True
elif torch_version.startswith("2.1."):
    is_torch_210 = True
elif torch_version.startswith("2.2."):
    is_torch_220 = True
elif torch_version.startswith("2.3.1"):
    is_torch_231 = True
elif torch_version.startswith("2.5.1"):
    is_torch_251 = True
else:
    raise ValueError(f"unsupported dicp torch version: {torch.__version__}")

is_torch_210_or_higher = version.parse(torch_version) >= version.parse("2.1")
is_torch_220_or_higher = version.parse(torch_version) >= version.parse("2.2")
is_torch_231_or_higher = version.parse(torch_version) >= version.parse("2.3.1")
is_torch_251_or_higher = version.parse(torch_version) >= version.parse("2.5.1")
