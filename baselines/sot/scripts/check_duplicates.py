import json
import sys
import os
from collections import defaultdict

def check_duplicates(input_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    print(f"Checking for duplicates in {input_file}...")
    
    question_counts = defaultdict(int)
    question_locations = defaultdict(list)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get('question')
                    if question:
                        question_counts[question] += 1
                        question_locations[question].append(i + 1)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {i+1}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    duplicates = {q: count for q, count in question_counts.items() if count > 1}
    
    if not duplicates:
        print("\nNo duplicate questions found.")
    else:
        print(f"\nFound {len(duplicates)} questions with duplicates:")
        print("-" * 50)
        for q, count in duplicates.items():
            locs = question_locations[q]
            # Print first 100 chars of question
            display_q = q[:100] + "..." if len(q) > 100 else q
            print(f"Question: \"{display_q}\"")
            print(f"  Count: {count}")
            print(f"  Line numbers: {locs}")
            print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_duplicates.py <extracted_results.jsonl>")
        sys.exit(1)
    check_duplicates(sys.argv[1])
