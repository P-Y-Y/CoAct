#!/usr/bin/env python3
"""
Extract the first N entries from a JSON file.

Usage:
  python extract_first_n.py <input_json> [--n 100] [--out <output_json>]

Behaviors:
- If top-level is a list: slice the first N items.
- If top-level is an object (dict):
    - If exactly one list-valued field exists, slice that field and keep others.
    - Otherwise, try common keys in order: ['data', 'items', 'questions'].
    - If none match, exit with an error explaining the keys.
"""

import argparse
import json
import sys
from pathlib import Path


def derive_default_out(in_path: Path, n: int) -> Path:
    stem = in_path.stem
    suffix = in_path.suffix or ".json"
    return in_path.with_name(f"{stem}_top{n}{suffix}")


def slice_list(lst, n):
    if not isinstance(lst, list):
        raise TypeError("slice_list expects a list")
    return lst[: max(0, int(n))]


def process(input_path: Path, out_path: Path, n: int) -> Path:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        subset = slice_list(data, n)
        to_write = subset
    elif isinstance(data, dict):
        # Collect list-valued keys
        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        target_key = None

        if len(list_keys) == 1:
            target_key = list_keys[0]
        else:
            for k in ["data", "items", "questions"]:
                if isinstance(data.get(k), list):
                    target_key = k
                    break

        if not target_key:
            print(
                "Error: could not determine which list field to slice. Available keys: ",
                list(data.keys()),
                file=sys.stderr,
            )
            sys.exit(2)

        # Copy and truncate the identified list
        to_write = dict(data)
        to_write[target_key] = slice_list(data[target_key], n)
    else:
        print("Error: unsupported top-level JSON type", type(data).__name__, file=sys.stderr)
        sys.exit(2)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_write, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract first N entries from JSON")
    parser.add_argument("input_json", type=str, help="Input JSON file path")
    parser.add_argument("--n", type=int, default=100, help="Number of entries to keep (default: 100)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    in_path = Path(args.input_json)
    if not in_path.exists():
        print(f"Error: input file does not exist: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else derive_default_out(in_path, args.n)

    try:
        result = process(in_path, out_path, args.n)
        print(f"Wrote: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
