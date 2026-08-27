"""GPU clock/temperature logging, and the rule for throwing a run away.

Why a benchmark harness needs this
----------------------------------
The measurements come from a laptop RTX 4050 in a chassis that cannot sustain
its boost clock. Left alone, a sweep produces a beautiful downward trend in
"performance" that is really just the GPU getting hot: the configs measured
last look slower than the ones measured first, whatever the kernels do. Two
defences, both used here:

  * the organizers' `benchmark_models` already **alternates** baseline and
    optimized rounds, so a drifting clock hits both sides roughly equally and
    largely cancels out of the ratio;
  * this module records the clock during each run so we can *prove* it stayed
    flat, and mechanically discard the runs where it did not.

The discard rule (from the sprint plan): compare the mean SM clock over the
timed window against the *opening* clock (mean of the first three samples in
that window). If the mean fell below 85% of the opening, the card throttled
mid-run and the row is marked `DISCARD:thermal`. The row is still written —
a discarded measurement is evidence about the hardware, and the report counts
them — it is just excluded from the summary statistics.

Everything here is a no-op without `nvidia-smi`, so the whole module imports
and runs on Person B's Mac; `ThermalLogger.available` is then False and
`summarize` is never reached.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

__all__ = [
    "ThermalLogger",
    "nvidia_smi_path",
    "parse_clock_log",
    "summarize",
    "wait_until_cool",
    "DISCARD_CLOCK_FRACTION",
]

#: A run is discarded when the mean SM clock over the timed window drops below
#: this fraction of the opening clock.
DISCARD_CLOCK_FRACTION = 0.85

#: Fields requested from nvidia-smi, in the order they appear in the CSV.
_QUERY_FIELDS = (
    "timestamp",
    "clocks.sm",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
)

_COLUMNS = ("timestamp", "sm_mhz", "temp_c", "power_w", "util_pct")


def nvidia_smi_path() -> Optional[str]:
    """Absolute path to nvidia-smi, or None on a machine without it."""
    return shutil.which("nvidia-smi")


class ThermalLogger:
    """Context manager that samples GPU clocks for the duration of a run.

    On a machine with no `nvidia-smi` (or when `enabled=False`) this does
    nothing at all: `available` is False, `path` is None, and `summarize()`
    returns None. Callers write empty CSV cells for the thermal columns.

    Usage::

        with ThermalLogger(...) as logger:
            logger.mark("timing_start")
            ...run the benchmark...
            logger.mark("timing_end")
        stats = logger.summarize(window=logger.window("timing_start", "timing_end"))
    """

    def __init__(
        self,
        log_dir: Path | str = "logs",
        git_sha: str = "nogit",
        strategy: str = "unknown",
        batch: int = 0,
        seq_len: int = 0,
        d_model: int = 0,
        enabled: bool = True,
        interval_s: int = 1,
        gpu_index: int = 0,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.git_sha = git_sha
        self.strategy = strategy
        self.batch = batch
        self.seq_len = seq_len
        self.d_model = d_model
        self.interval_s = interval_s
        self.gpu_index = gpu_index

        self._executable = nvidia_smi_path() if enabled else None
        self.available = self._executable is not None
        self.path: Optional[Path] = None
        self.reason: str = "" if self.available else (
            "disabled" if not enabled else "nvidia-smi not found"
        )

        self._process: Optional[subprocess.Popen] = None
        self._handle = None
        self._t0: Optional[float] = None
        self._marks: Dict[str, float] = {}

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "ThermalLogger":
        if not self.available:
            return self
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M")
        name = (
            f"clocks_{self.git_sha}_{stamp}_{self.strategy}_"
            f"{self.batch}x{self.seq_len}x{self.d_model}.csv"
        )
        self.path = self.log_dir / name
        try:
            self._handle = self.path.open("w")
            self._process = subprocess.Popen(
                [
                    self._executable,
                    f"--id={self.gpu_index}",
                    f"--query-gpu={','.join(_QUERY_FIELDS)}",
                    "--format=csv",
                    "-l",
                    str(self.interval_s),
                ],
                stdout=self._handle,
                stderr=subprocess.DEVNULL,
            )
            self._t0 = time.time()
        except OSError as exc:  # spawning failed; carry on without thermal data
            self.available = False
            self.reason = f"could not start nvidia-smi: {exc}"
            self._close_handle()
            self.path = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None
        self._close_handle()

    def _close_handle(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            finally:
                self._handle = None

    # -- windowing ---------------------------------------------------------

    def mark(self, label: str) -> None:
        """Record a named instant, in seconds since logging started.

        Used to restrict `summarize` to the *timed* window, so warmup and
        compilation do not pollute the clock statistics.
        """
        if self._t0 is not None:
            self._marks[label] = time.time() - self._t0

    def window(self, start: str, end: str) -> Optional[Tuple[float, float]]:
        if start in self._marks and end in self._marks:
            return self._marks[start], self._marks[end]
        return None

    # -- results -----------------------------------------------------------

    def summarize(self, window: Optional[Tuple[float, float]] = None) -> Optional[Dict]:
        """Parse this run's log and apply the discard rule. None if unavailable."""
        if not self.available or self.path is None or not self.path.exists():
            return None
        try:
            frame = parse_clock_log(self.path)
        except (ValueError, OSError, pd.errors.ParserError):
            return None
        if frame.empty:
            return None
        return summarize(frame, window=window)


def parse_clock_log(path: Path | str) -> pd.DataFrame:
    """Read an nvidia-smi `--format=csv` clock log into a tidy frame.

    Returns columns `t` (seconds from the first sample), `sm_mhz`, `temp_c`,
    `power_w`, `util_pct`. nvidia-smi writes units into both the header
    (`clocks.current.sm [MHz]`) and the values (`2400 MHz`), and emits
    `[N/A]` for fields the driver will not report; all three are stripped here.
    """
    frame = pd.read_csv(path, skipinitialspace=True)
    if frame.shape[1] < len(_COLUMNS):
        raise ValueError(
            f"{path}: expected {len(_COLUMNS)} columns "
            f"({', '.join(_COLUMNS)}), found {frame.shape[1]}"
        )
    frame = frame.iloc[:, : len(_COLUMNS)]
    frame.columns = list(_COLUMNS)

    # Repeated header rows appear if nvidia-smi is restarted into the same file.
    frame = frame[frame["timestamp"].astype(str).str.strip().str.lower() != "timestamp"]

    for column in ("sm_mhz", "temp_c", "power_w", "util_pct"):
        frame[column] = pd.to_numeric(
            frame[column]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", None),
            errors="coerce",
        )

    stamps = pd.to_datetime(
        frame["timestamp"].astype(str).str.strip(),
        format="%Y/%m/%d %H:%M:%S.%f",
        errors="coerce",
    )
    if stamps.notna().any():
        frame["t"] = (stamps - stamps.min()).dt.total_seconds()
    else:
        # No parseable timestamps: fall back to sample index as seconds.
        frame["t"] = range(len(frame))

    frame = frame.dropna(subset=["sm_mhz"]).reset_index(drop=True)
    return frame[["t", "sm_mhz", "temp_c", "power_w", "util_pct"]]


def summarize(
    frame: pd.DataFrame,
    window: Optional[Tuple[float, float]] = None,
) -> Dict:
    """Clock statistics over `window`, plus the discard verdict.

    `opening_sm_mhz` is the mean of the first three samples in the window —
    three rather than one because a single sample lands wherever the 1 Hz
    poller happened to fire and is noisy.
    """
    if window is not None:
        start, end = window
        selected = frame[(frame["t"] >= start) & (frame["t"] <= end)]
        if len(selected) < 3:
            # The window was shorter than a few polling intervals; a verdict
            # from one or two samples would be noise. Use everything instead.
            selected = frame
    else:
        selected = frame

    if selected.empty:
        return {
            "mean_sm_clock_mhz": None,
            "max_temp_c": None,
            "opening_sm_mhz": None,
            "discard": False,
            "n_samples": 0,
        }

    mean_clock = float(selected["sm_mhz"].mean())
    opening = float(selected["sm_mhz"].head(3).mean())
    max_temp = selected["temp_c"].max()

    return {
        "mean_sm_clock_mhz": mean_clock,
        "max_temp_c": None if pd.isna(max_temp) else float(max_temp),
        "opening_sm_mhz": opening,
        "discard": bool(opening > 0 and mean_clock < DISCARD_CLOCK_FRACTION * opening),
        "n_samples": int(len(selected)),
    }


def wait_until_cool(
    max_temp_c: float = 45.0,
    timeout_s: float = 120.0,
    poll_s: float = 5.0,
    gpu_index: int = 0,
) -> Optional[float]:
    """Block until the GPU is below `max_temp_c`. No-op without nvidia-smi.

    Returns the last temperature read, or None if it could not be read at all.
    Times out rather than blocking a sweep forever — a card that will not cool
    below the threshold is itself worth recording.
    """
    executable = nvidia_smi_path()
    if executable is None:
        return None

    deadline = time.time() + timeout_s
    temperature: Optional[float] = None
    while True:
        try:
            completed = subprocess.run(
                [
                    executable,
                    f"--id={gpu_index}",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            temperature = float(completed.stdout.strip().splitlines()[0])
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            return temperature
        if temperature <= max_temp_c or time.time() >= deadline:
            return temperature
        time.sleep(poll_s)
