#!/usr/bin/env python3
"""List every unfilled placeholder in the submission documents.

    python docs/check_ready.py

Exists because the realistic failure mode at 6pm on submission day is not a bug
— it is shipping a report with the word TODO in it. Unfinished slots are written
as `<FILL: what is missing>`; this finds them all and exits non-zero while any
remain, so it can gate a submission checklist.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

DOCS = ("TECH_REPORT.md", "DEVPOST.md", "VIDEO_SCRIPT.md", "README.md")

#: `<FILL: description>`, or the older owner-tagged form still in the plan.
PATTERN = re.compile(r"<FILL\s*(A/B|A|B)?\s*:?\s*(.*?)>", re.IGNORECASE | re.DOTALL)

#: The README uses a different marker for the same idea.
TODO_PATTERN = re.compile(r"_TODO\b(.*?)_", re.DOTALL)


def scan(root: Path) -> Dict[str, List[Tuple[int, str, str]]]:
    """{relative path: [(line number, owner, description), ...]}

    Scans the whole file rather than line by line. Placeholders wrap: a slot
    written as

        `<FILL: worst max_abs_err across all passing runs, and the
        margin to each threshold>`

    has no line containing both the opening and the closing marker, so a
    per-line scan silently misses it. That is the worst possible failure for
    this script — it reports "ready" while real gaps remain, which is the one
    outcome it exists to prevent.
    """
    found: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for name in DOCS:
        path = root / "docs" / name if name != "README.md" else root / name
        if not path.exists():
            continue
        text = path.read_text()
        for match in PATTERN.finditer(text):
            owner, description = match.group(1), match.group(2)
            if description.strip() in ("\u2026", "...", ""):
                continue
            number = text.count("\n", 0, match.start()) + 1
            found[str(path.relative_to(root))].append(
                (number, (owner or "?").upper(), " ".join(description.split())[:90])
            )
        for match in TODO_PATTERN.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            found[str(path.relative_to(root))].append(
                (number, "?", "TODO" + " ".join(match.group(1).split())[:86])
            )
    return dict(found)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python docs/check_ready.py")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)

    root = Path(args.root)
    found = scan(root)

    wanted = None
    total = 0
    by_owner: Dict[str, int] = defaultdict(int)

    for path, entries in sorted(found.items()):
        rows = [e for e in entries if wanted is None or e[1] == wanted]
        if not rows:
            continue
        print(f"\n{path} — {len(rows)} outstanding")
        for number, owner, description in rows:
            print(f"  {path}:{number}  {description}")
            by_owner[owner] += 1
            total += 1

    print()
    if total == 0:
        print("No placeholders remain. Documents are ready to submit.")
        return 0

    print(f"{total} placeholder(s) outstanding. Fill them before submitting.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
