#!/usr/bin/env python3
"""List every unfilled placeholder in the submission documents.

    python docs/check_ready.py            # report, exit 1 if anything is missing
    python docs/check_ready.py --owner A  # only what Person A still owes

Exists because the realistic failure mode at 6pm on submission day is not a bug
— it is shipping a report with the word TODO in it. Placeholders are written as
`<FILL A: …>` / `<FILL B: …>` / `<FILL A/B: …>`; this finds them, groups them by
who owes them, and exits non-zero while any remain, so it can gate a submission
checklist.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

DOCS = ("TECH_REPORT.md", "DEVPOST.md", "VIDEO_SCRIPT.md", "README.md")

#: `<FILL A: description>` — the owner is A, B, or A/B.
PATTERN = re.compile(r"<FILL\s*(A/B|A|B)?\s*:?\s*(.*?)>", re.IGNORECASE | re.DOTALL)

#: The README uses a different marker for the same idea.
TODO_PATTERN = re.compile(r"_TODO\b(.*?)_", re.DOTALL)


def scan(root: Path) -> Dict[str, List[Tuple[int, str, str]]]:
    """{relative path: [(line number, owner, description), ...]}"""
    found: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for name in DOCS:
        path = root / "docs" / name if name != "README.md" else root / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for owner, description in PATTERN.findall(line):
                # `<FILL A: …>` with a literal ellipsis is the *legend* — the
                # sentence in each document explaining the convention — not a
                # slot anyone has to fill. Counting it means this script can
                # never reach zero, which would make its exit code useless as a
                # submission gate.
                if description.strip() in ("…", "...", ""):
                    continue
                found[str(path.relative_to(root))].append(
                    (number, (owner or "?").upper(), " ".join(description.split())[:90])
                )
            for description in TODO_PATTERN.findall(line):
                found[str(path.relative_to(root))].append(
                    (number, "?", "TODO" + " ".join(description.split())[:86])
                )
    return dict(found)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python docs/check_ready.py")
    parser.add_argument("--owner", choices=("A", "B", "A/B"),
                        help="show only placeholders owed by this person")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)

    root = Path(args.root)
    found = scan(root)

    wanted = args.owner.upper() if args.owner else None
    total = 0
    by_owner: Dict[str, int] = defaultdict(int)

    for path, entries in sorted(found.items()):
        rows = [e for e in entries if wanted is None or e[1] == wanted]
        if not rows:
            continue
        print(f"\n{path} — {len(rows)} outstanding")
        for number, owner, description in rows:
            print(f"  {path}:{number}  [{owner:<3}] {description}")
            by_owner[owner] += 1
            total += 1

    print()
    if total == 0:
        scope = f" for {wanted}" if wanted else ""
        print(f"No placeholders remain{scope}. Documents are ready to submit.")
        return 0

    breakdown = ", ".join(f"{owner}: {count}" for owner, count in sorted(by_owner.items()))
    print(f"{total} placeholder(s) outstanding ({breakdown}).")
    print("Fill them, or run with --owner to see one person's share.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
