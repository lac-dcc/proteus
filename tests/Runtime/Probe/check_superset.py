#!/usr/bin/env python3
import re
import sys

WORDS_RE = re.compile(r"words = array<i64:\s*(-?\d+)\s*>")


def extract_words(path):
    with open(path) as f:
        return [int(w) for w in WORDS_RE.findall(f.read())]


def main():
    predicted_words = extract_words(sys.argv[1])
    actual_words = extract_words(sys.argv[2])

    ok = True
    for i, (predicted, actual) in enumerate(zip(predicted_words, actual_words)):
        unproven = actual & ~predicted
        if unproven != 0:
            print(f"soundness violation in word {i}: predicted={predicted:#x} "
                  f"actual={actual:#x} dense-in-actual-but-not-predicted="
                  f"{unproven:#x}", file=sys.stderr)
            ok = False

    if not ok:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
