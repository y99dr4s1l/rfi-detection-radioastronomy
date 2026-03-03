import time


class Timer:
    """Simple context manager for timing code blocks."""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start

    def __str__(self):
        return f'{self.elapsed:.4f}s'