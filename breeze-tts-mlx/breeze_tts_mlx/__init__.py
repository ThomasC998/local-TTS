"""Apple-Silicon MLX inference support for Breeze TTS 2.

MLX is imported lazily by the implementation modules so importing this package on
Linux or on a non-MLX Python environment remains harmless.
"""

from .config import BreezeMLXConfig

__all__ = ["BreezeMLXConfig"]
