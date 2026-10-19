"""Keep only the results whose question is among the first N questions of a dataset.

Usage:
  python scripts/extract_top_n_results.py --dataset data/hotpot/x.json \
      --results results/.../extracted_results.jsonl --output results/.../top_200_results.jsonl [--n 200]
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Source dataset JSON (list of items with a 'question' field)")
    parser.add_argument("--results", required=True, help="extracted_results.jsonl produced by sot/main.py")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--n", type=int, default=200, help="Number of leading dataset questions to keep (default: 200)")
    args = parser.parse_args()

    # 1. Read the first N questions from the source dataset
    print(f"Reading source dataset: {args.dataset}")
    with open(args.dataset, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    target_entries = source_data[:args.n]
    target_questions = set()

    for entry in target_entries:
        if "question" in entry:
            target_questions.add(entry["question"].strip())

    print(f"Targeting {len(target_questions)} unique questions from the first {args.n} entries.")

    # 2. Scan results file and match
    print(f"Scanning results file: {args.results}")
    matched_results = []

    if not os.path.exists(args.results):
        print(f"Error: Results file not found at {args.results}")
        return

    with open(args.results, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                result = json.loads(line)
                q = result.get("question", "").strip()

                if q in target_questions:
                    matched_results.append(result)
            except json.JSONDecodeError:
                continue

    # 3. Save matching results
    print(f"Found {len(matched_results)} matches.")
    print(f"Saving to {args.output}...")

    with open(args.output, 'w', encoding='utf-8') as f:
        for res in matched_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
