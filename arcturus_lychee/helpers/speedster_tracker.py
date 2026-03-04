import time

class SpeedTimer:
    def __init__(self):
        self._start = None
        self._end   = None
        self.start()

    def start(self):
        """Start the timer."""
        self._start = time.perf_counter()
        self._end = None

    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self._start is None:
            raise RuntimeError("Timer has not been started.")
        self._end = time.perf_counter()
        return self.elapsed()

    def elapsed(self) -> float:
        """Return elapsed time in seconds without stopping the timer."""
        if self._start is None:
            raise RuntimeError("Timer has not been started.")
        end_time = self._end if self._end is not None else time.perf_counter()
        return end_time - self._start

    @staticmethod
    def average(timers: list["SpeedTimer"]) -> float:
        """
        Return the average elapsed time among timers that have been stopped.
        Timers with _end = None are skipped.
        Returns None if no timers are valid.
        """
        elapsed_times = [t.elapsed() for t in timers if t._end is not None]

        if not elapsed_times:
            return float('nan')
        
        return sum(elapsed_times) / len(elapsed_times)
    
    def formatted_elapsed(self) -> str:
        """Return elapsed time in 'H hours M minutes S.SSS seconds' format."""
        total_seconds = self.elapsed()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        parts = []
        if hours > 0:
            parts.append(f"{hours} hours")
        if minutes > 0:
            parts.append(f"{minutes} minutes")
        parts.append(f"{seconds:.2f} seconds")  # Always include seconds

        return " ".join(parts)

    @staticmethod
    def total_span(timers: list["SpeedTimer"]) -> "SpeedTimer":
        """
        Return a new SpeedTimer object with _start = earliest start
        and _end = latest end among the given timers.
        """
        # Filter out timers that haven't started or stopped
        valid_timers = [t for t in timers if t._start is not None and t._end is not None]
        if not valid_timers:
            raise ValueError("No valid timers with start and stop times.")

        earliest_start = min(t._start for t in valid_timers)
        latest_end = max(t._end for t in valid_timers)

        # Create a new SpeedTimer object with custom start/end
        total_timer = SpeedTimer.__new__(SpeedTimer)  # bypass __init__
        total_timer._start = earliest_start
        total_timer._end = latest_end
        return total_timer

    @staticmethod
    def estimate_time(speed_timer : "SpeedTimer", amount : int) -> str:

        # get total time
        estimated_seconds = speed_timer.elapsed() * amount

        # convert to days, hours, minutes...
        days,  rem       = divmod(estimated_seconds, 86400)
        hours, rem       = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)

        # create the parts
        parts = []
        
        # day (insure int)
        days = int(days)
        if days > 0:
            parts.append(f"{days} Days")

        # hours
        hours = int(hours)
        if hours > 0:
            parts.append(f"{hours} Hours")

        # min
        minutes = int(minutes)
        if minutes > 0:
            parts.append(f"{minutes} Minutes")
        
        # Always include seconds
        seconds = round(seconds)
        seconds = int(seconds)
        parts.append(f"{seconds} Seconds")  

        # join the message
        message =  " ".join(parts)
        return f"Estimated remaning time : {message}"