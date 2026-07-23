from __future__ import annotations
import time
class ProgressTimer:
    def __init__(self): self.started = time.time()
    @property
    def elapsed_seconds(self): return time.time() - self.started
    def elapsed_text(self): return f"{self.elapsed_seconds/60:.1f} min"
