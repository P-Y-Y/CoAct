import json
import sys
import os

def check_missing(file_path, start=0, end=823):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    found_indices = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'index' in data:
                        found_indices.add(data['index'])
                except json.JSONDecodeError:
                    # Try to handle likely corrupted lines if needed, or just skip
                    pass
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    expected_indices = set(range(start, end + 1))
    missing_indices = sorted(list(expected_indices - found_indices))
    
    print(f"File: {file_path}")
    print(f"Total found unique indices: {len(found_indices)}")
    print(f"Total expected indices (0-{end}): {len(expected_indices)}")
    print(f"Missing count: {len(missing_indices)}")
    
    if missing_indices:
        print(f"Missing indices ({len(missing_indices)}):")
        # Print in a compact way
        print(missing_indices)
    else:
        print("No missing indices.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_missing.py <jsonl_file> [max_index]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    max_idx = 823
    if len(sys.argv) > 2:
        max_idx = int(sys.argv[2])
        
    check_missing(file_path, 0, max_idx)
