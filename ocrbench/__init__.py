"""ocrbench: benchmark PaddleOCR vs Google Document AI on handwritten exam booklets."""

__version__ = "0.1.0"

from .config import load_config  # noqa: E402

__all__ = ["load_config", "__version__"]
