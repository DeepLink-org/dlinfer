import torch
from packaging import version

torch_version = torch.__version__
is_torch_200 = False
is_torch_210_or_higher = False
is_torch_220_or_higher = False
is_torch_230_or_higher = False

if torch_version.startswith("2.0"):
    is_torch_200 = True
elif version.parse(torch_version) >= version.parse("2.1"):
    is_torch_210_or_higher = True
else:
    raise ValueError(f"unsupported dicp torch version: {torch.__version__}")

if version.parse(torch_version) >= version.parse("2.2"):
    is_torch_220_or_higher = True
if version.parse(torch_version) >= version.parse("2.3"):
    is_torch_230_or_higher = True
