from __future__ import annotations

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="replace first two line")

parser.add_argument("files", type=str, nargs="*")
parsed_args = parser.parse_args()

n_files = 0
for rest_file in parsed_args.files:
    with Path(rest_file).open() as f:
        first_word = f.readline().strip().split()[0]
        if "." not in first_word:
            continue
        replacement = first_word.split(".")[-1]
        underline = f.readline()[0] * len(replacement)
        lower_file = f.read()
    with Path(rest_file).open("w") as f:
        f.write("\n".join((replacement, underline, lower_file)))
    n_files += 1
