"""training package — visual + audio training loops."""
from .train_visual import train_model

# Audio exports — populated when train_audio.py is implemented.
# from .train_audio import ...

__all__ = ["train_model"]
