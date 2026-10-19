import json
import sys
import os
from collections import defaultdict

def remove_duplicates(input_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    original_count = len(lines)
    
    # Store entries as: { question_string: [ (line_index, json_data, original_line_string), ... ] }
    groups = defaultdict(list)
    
    for i, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue
        try:
            data = json.loads(line_str)
            question = data.get('question')
            
            # Using None key for entries without a question field
            groups[question].append((i, data, line))
                
        except json.JSONDecodeError:
            print(f"Warning: Skipping invalid JSON at line {i+1}")

    final_entries = []

    for question, entries in groups.items():
        if question is None:
            # If 'question' key is missing, we don't treat them as duplicates of each other
            # just keep them all.
            final_entries.extend(entries)
            continue
            
        if len(entries) == 1:
            final_entries.append(entries[0])
        else:
            # Multiple entries for this question.
            # Priority:
            # 1. Answer != "0" (and pick the first one of these)
            # 2. Fallback: First one that appears (if all are "0")
            
            def is_valid_answer(data):
                ans = data.get('answer')
                # Check for string "0" or int 0
                return str(ans).strip() != "0"
            
            # Filter candidates that have non-0 answers
            valid_candidates = [e for e in entries if is_valid_answer(e[1])]
            
            if valid_candidates:
                # If we have valid answers, keep the first one found in the file
                final_entries.append(valid_candidates[0])
            else:
                # If all answers are "0", keep the first one found in the file
                final_entries.append(entries[0])

    # Sort final list by original line index to preserve original file order
    final_entries.sort(key=lambda x: x[0])
    
    output_temp = input_file + ".tmp"
    with open(output_temp, 'w', encoding='utf-8') as f:
        for _, _, line_content in final_entries:
            # Use original line content including newline if present
            f.write(line_content)
            # If original line was stripped or manipulated, ensure newline
            if not line_content.endswith('\n'):
                f.write('\n')
            
    backup_file = input_file + ".bak"
    
    if os.path.exists(backup_file):
        os.remove(backup_file)
    os.rename(input_file, backup_file)
    os.rename(output_temp, input_file)
    
    print(f"Done. Original lines: {original_count}, New lines: {len(final_entries)}")
    print(f"Original file backed up to {backup_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_duplicates.py <extracted_results.jsonl>")
        sys.exit(1)
    remove_duplicates(sys.argv[1])
