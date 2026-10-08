"""
Compatibility shim: `torch._six` was removed in PyTorch 2.0, but timm 0.3.2
(the version the STP code pins) imports `container_abcs` from it.

Import this before timm if you are on a modern PyTorch:

    import tools.torch_six_compat  # noqa: F401
    import timm

Nothing is monkey-patched beyond recreating the one attribute timm asks for.
"""
import sys
import types
from collections import abc as container_abcs

if 'torch._six' not in sys.modules:
    try:
        import torch._six  # noqa: F401
    except ModuleNotFoundError:
        mod = types.ModuleType('torch._six')
        mod.container_abcs = container_abcs
        mod.string_classes = (str, bytes)
        mod.int_classes = (int,)
        mod.file_system = None
        mod.inf = float('inf')
        mod.nan = float('nan')
        sys.modules['torch._six'] = mod
