"""Extract the results that correspond to the first N questions of a dataset.

Usage:
  python extract_top_n_results.py --dataset data/hotpot/hard-level-questions-500_validation.json \
      --results results/hotpot/glm/answer_..._backup.jsonl \
      --output results/hotpot/glm/top100_extracted_results.jsonl [--n 100]
"""
import argparse
import json
import os


def main(dataset_path, result_file_path, output_file_path, n):
    print(f"Step 1: Loading dataset from {dataset_path} ...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Take the first N questions, keeping the dataset order
    target_questions_list = []
    for i, item in enumerate(dataset[:n]):
        q = item.get("question", "").strip()
        target_questions_list.append(q)

    print(f"Target count: {len(target_questions_list)} questions (from the first {n} items of the dataset).")

    print(f"Step 2: Loading result file from {result_file_path} ...")
    if not os.path.exists(result_file_path):
        print(f"Error: Result file not found: {result_file_path}")
        return

    # Lookup table: problem -> result entry
    results_map = {}
    with open(result_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # The question field in result files is "problem"
                if "problem" in entry:
                    problem_text = entry["problem"].strip()
                    results_map[problem_text] = entry
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON at line {line_num}")
                continue

    print(f"Loaded {len(results_map)} unique entries from result file.")

    print("Step 3: Matching and Extracting...")
    matched_entries = []
    found_count = 0
    missing_count = 0

    # Extract in the order of the first N dataset questions
    for q in target_questions_list:
        if q in results_map:
            matched_entries.append(results_map[q])
            found_count += 1
        else:
            missing_count += 1

    print(f"Summary -> Matched: {found_count}, Missing: {missing_count}")

    output_abs_path = os.path.abspath(output_file_path)
    print(f"Step 4: Saving extracted results to {output_abs_path} ...")
    with open(output_abs_path, 'w', encoding='utf-8') as f:
        for entry in matched_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract results matching the first N dataset questions.")
    parser.add_argument("--dataset", required=True, help="Dataset JSON (list of records with a 'question' field)")
    parser.add_argument("--results", required=True, help="Result JSONL file (records with a 'problem' field)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--n", type=int, default=100, help="Number of leading dataset questions (default: 100)")
    args = parser.parse_args()
    main(args.dataset, args.results, args.output, args.n)
