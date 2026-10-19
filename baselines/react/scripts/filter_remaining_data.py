"""Write the dataset records that do not yet appear in a (partial) result file.

Useful for resuming an interrupted run:
  python filter_remaining_data.py --processed results/.../answer_..._backup.jsonl \
      --dataset data/hotpot/hard-level-questions-500_validation.json \
      --output data/hotpot/hard-level-questions-remaining.json
"""
import argparse
import json
import os


def filter_dataset(processed_file, full_dataset_file, output_file):
    # 1. Load processed questions
    processed_questions = set()
    print(f"Reading processed file: {processed_file}")
    
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # The key in the processed file is "problem"
                    if "problem" in data:
                        processed_questions.add(data["problem"].strip())
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
    else:
        print(f"Warning: Processed file not found at {processed_file}. Assuming no questions processed.")

    print(f"Found {len(processed_questions)} processed questions.")

    # 2. Load full dataset
    print(f"Reading full dataset: {full_dataset_file}")
    with open(full_dataset_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    print(f"Total questions in full dataset: {len(full_data)}")

    # 3. Filter data
    remaining_data = []
    
    for item in full_data:
        # The key in the full dataset is "question"
        question_text = item.get("question", "").strip()
        
        if question_text not in processed_questions:
            remaining_data.append(item)

    print(f"Remaining questions to process: {len(remaining_data)}")

    # 4. Save result
    print(f"Saving remaining data to: {output_file}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(remaining_data, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out dataset records that were already processed.")
    parser.add_argument("--processed", required=True, help="Result JSONL with a 'problem' field per line")
    parser.add_argument("--dataset", required=True, help="Full dataset JSON (list of records with a 'question' field)")
    parser.add_argument("--output", required=True, help="Output JSON with the remaining records")
    args = parser.parse_args()
    filter_dataset(args.processed, args.dataset, args.output)
