#!/usr/bin/env python3
"""
sample_random.py

Randomly sample N records (default 500) from an input JSON file and write them out.
Usage:
  python sample_random.py --input /path/to/in.json --output /path/to/out.json --n 500 --seed 42
"""
import argparse
import json
import random
import sys
from copy import deepcopy


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_list_root(obj):
    """If top-level is a list, return (None, list).
    If top-level is a dict and has one or more list values, return (key, list) for the first found.
    Otherwise return (None, None).
    """
    if isinstance(obj, list):
        return None, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                return k, v
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input JSON file path')
    p.add_argument('--output', '-o', help='Output JSON file path (default: input name with -N.json suffix)')
    p.add_argument('--n', '-n', type=int, default=500, help='Number of records to sample (default 500)')
    p.add_argument('--seed', type=int, default=None, help='Random seed (optional)')
    args = p.parse_args()

    data = load_json(args.input)
    key, lst = find_list_root(data)
    if lst is None:
        print('No list found in the input JSON; cannot sample.', file=sys.stderr)
        sys.exit(2)

    if args.seed is not None:
        random.seed(args.seed)

    count = min(args.n, len(lst))
    sampled = random.sample(lst, count)

    if key is None:
        out_obj = sampled
    else:
        out_obj = deepcopy(data)
        out_obj[key] = sampled

    out_path = args.output or args.input.replace('.json', f'-{count}.json')
    save_json(out_obj, out_path)
    print(f'Saved {count} records to {out_path}')


if __name__ == '__main__':
    main()
