import time
from typing import Optional


class SpeedTimer:
    """A timer for one epoch. It also estimates the time to the end of the training."""

    def __init__(self) -> None:
        self._start : float           = time.perf_counter()
        self._end   : Optional[float] = None

    def start(self) -> None:
        """Start the timer again."""
        self._start = time.perf_counter()
        self._end   = None

    def stop(self) -> float:
        """Stop the timer, and return the time in seconds."""
        self._end = time.perf_counter()
        return self.elapsed()

    def elapsed(self) -> float:
        """Return the time in seconds. If the timer did not stop, the time goes to this moment."""
        end_time = self._end if self._end is not None else time.perf_counter()
        return end_time - self._start

    @staticmethod
    def estimate_time(speed_timer : "SpeedTimer", amount : int) -> str:
        """Return a text with the estimated time for 'amount' more epochs of the same length."""
        estimated_seconds = speed_timer.elapsed() * amount

        days,    rest    = divmod(estimated_seconds, 86400)
        hours,   rest    = divmod(rest, 3600)
        minutes, seconds = divmod(rest, 60)

        parts = []
        if days >= 1:
            parts.append(f"{int(days)} days")
        if hours >= 1:
            parts.append(f"{int(hours)} hours")
        if minutes >= 1:
            parts.append(f"{int(minutes)} minutes")
        parts.append(f"{int(round(seconds))} seconds")

        return f"Estimated remaining time: {' '.join(parts)}"
