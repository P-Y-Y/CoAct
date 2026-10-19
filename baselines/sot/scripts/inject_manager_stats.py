import csv
import json
import argparse
import os
import shutil
import sys

# Increase CSV field limit to handle large JSON strings
csv.field_size_limit(sys.maxsize)

def load_csv_data(csv_path):
    """
    Load manager_stats from a CSV file.
    Returns a dict mapping question/request -> manager_stats dict.
    """
    manager_stats_map = {}
    print(f"Reading CSV from: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The question field may be called 'request' in the CSV
            question = row.get('request') or row.get('question')
            if not question:
                continue
            
            # Parse manager_stats
            # In the CSV, dicts are usually stored as JSON strings
            ms_str = row.get('manager_stats')
            if ms_str:
                try:
                    # The CSV may escape quotes multiple times, or be plain JSON
                    # ast.literal_eval would also work, but json.loads is more standard
                    manager_stats = json.loads(ms_str)
                    manager_stats_map[question.strip()] = manager_stats
                except json.JSONDecodeError:
                    # Fall back to Python-style repr (single quotes)
                    try:
                        import ast
                        manager_stats = ast.literal_eval(ms_str)
                        manager_stats_map[question.strip()] = manager_stats
                    except:
                        print(f"Warning: Failed to parse manager_stats for question: {question[:30]}...")
    
    print(f"Loaded {len(manager_stats_map)} manager_stats entries from CSV.")
    return manager_stats_map

def update_jsonl(jsonl_path, manager_stats_map, output_path=None):
    """
    Read the JSONL, inject manager_stats, and write the file back.
    """
    if output_path is None:
        output_path = jsonl_path + ".updated"
    
    print(f"Updating JSONL data...")
    updated_count = 0
    total_count = 0
    
    new_records = []
    
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    total_count += 1
                    
                    q = record.get('question')
                    if q:
                        q_key = q.strip()
                        # Look up the matching manager_stats
                        if q_key in manager_stats_map:
                            if 'manager_stats' not in record or not record['manager_stats']:
                                record['manager_stats'] = manager_stats_map[q_key]
                                updated_count += 1
                        else:
                            # CSV and JSONL questions may differ slightly (e.g. whitespace); both are .strip()-ed
                            pass
                    
                    new_records.append(record)
                except json.JSONDecodeError:
                    print("Warning: Skipping invalid JSON line.")
    
    # Write the new file
    print(f"Writing updated data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"Process complete. Total records: {total_count}, Updated with manager_stats: {updated_count}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Inject manager_stats from CSV to JSONL")
    parser.add_argument("--csv", required=True, help="Path to the source CSV file")
    parser.add_argument("--jsonl", required=True, help="Path to the target JSONL file")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the original JSONL file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    if not os.path.exists(args.jsonl):
        print(f"Error: JSONL file not found: {args.jsonl}")
        return

    # 1. Load data from CSV
    manager_stats_map = load_csv_data(args.csv)
    
    # 2. Update JSONL
    new_file = update_jsonl(args.jsonl, manager_stats_map)
    
    # 3. Handle replacement if requested
    if args.inplace:
        print(f"Overwriting original file: {args.jsonl}")
        shutil.move(new_file, args.jsonl)
    else:
        print(f"New file created at: {new_file}")

if __name__ == "__main__":
    main()
