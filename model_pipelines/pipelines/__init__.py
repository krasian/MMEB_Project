"""pipelines package — visual, audio, and multimodal pipeline runners."""
from .run_visual import run_full_pipeline, run_evaluation

# Audio + multimodal exports — populated when those pipelines are implemented.
# from .run_audio import ...
# from .run_multimodal import ...

__all__ = [
    "run_full_pipeline",
    "run_evaluation",
]
