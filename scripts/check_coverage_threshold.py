#!/usr/bin/env python3

import json
import sys


def main() -> int:
    threshold = float(sys.argv[1])
    report = json.load(sys.stdin)
    lines = report["data"][0]["totals"]["lines"]
    percent, covered, count = lines["percent"], lines["covered"], lines["count"]

    print(
        f"Line coverage: {percent:.2f}% ({covered}/{count} lines), "
        f"threshold: {threshold:.2f}%"
    )

    if percent < threshold:
        print("FAIL: coverage is below threshold", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
