"""Generate synthetic nvidia-smi clock logs so the discard rule can be tested
on a machine that has no GPU.

Run:  python tests/fixtures/make_clock_fixtures.py

Two fixtures, in the exact shape `nvidia-smi --format=csv` emits (units in both
the header and the values, because that is what the real tool does and the
parser has to cope with it):

  clocks_synthetic.csv   a card that boosts to 2400 MHz, throttles hard after a
                         few seconds and settles near 1600 MHz. summarize()
                         must flag discard=True.
  clocks_flat.csv        a card that holds its clock. discard=False.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

HEADER = (
    "timestamp, clocks.current.sm [MHz], temperature.gpu, "
    "power.draw [W], utilization.gpu [%]"
)
START = datetime(2026, 8, 27, 21, 30, 0)


def _row(index: int, mhz: int, temp: int, watts: float, util: int) -> str:
    stamp = (START + timedelta(seconds=index)).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    return f"{stamp}, {mhz} MHz, {temp}, {watts:.2f} W, {util} %"


def write_throttling(path: Path, samples: int = 30, seed: int = 7) -> None:
    """Boost clock for 5 s, then throttle to ~1600 MHz and stay there."""
    rng = random.Random(seed)
    lines = [HEADER]
    for index in range(samples):
        if index < 5:
            mhz = 2400 + rng.randint(-15, 15)
            temp = 52 + index
            watts = 78.0 + rng.uniform(-2, 2)
        else:
            # Fast decay into a plateau, the shape a power/thermal cap produces.
            decay = min(1.0, (index - 5) / 4.0)
            mhz = int(2400 - 780 * decay) + rng.randint(-20, 20)
            temp = min(87, 57 + (index - 5))
            watts = 60.0 + rng.uniform(-3, 3)
        lines.append(_row(index, mhz, temp, watts, rng.randint(96, 100)))
    path.write_text("\n".join(lines) + "\n")


def write_flat(path: Path, samples: int = 30, seed: int = 11) -> None:
    """A card that holds 2400 MHz throughout. Nothing to discard."""
    rng = random.Random(seed)
    lines = [HEADER]
    for index in range(samples):
        lines.append(
            _row(
                index,
                2400 + rng.randint(-12, 12),
                55 + rng.randint(0, 3),
                76.0 + rng.uniform(-2, 2),
                rng.randint(97, 100),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    here = Path(__file__).parent
    write_throttling(here / "clocks_synthetic.csv")
    write_flat(here / "clocks_flat.csv")
    print(f"wrote {here / 'clocks_synthetic.csv'}")
    print(f"wrote {here / 'clocks_flat.csv'}")


if __name__ == "__main__":
    main()
