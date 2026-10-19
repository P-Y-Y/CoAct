#!/usr/bin/env python3
"""
Download and decrypt the xbench DeepSearch dataset.

1. Load "xbench/DeepSearch-2510" from Hugging Face.
2. Convert the Parquet records to JSON.
3. Decrypt the `prompt` / `answer` fields (XOR with the per-record canary, as in xbench-evals).

Usage:
  python convert_and_decrypt_dataset.py [--output data/deepsearch/DeepSearch-2510_decrypted.json]
"""

import argparse
import base64
import json
import sys
import os
from datasets import load_dataset

def xor_decrypt(data, key):
    """
    XOR decrypt data with a key
    (copied from xbench_evals.py)
    """
    key_bytes = key.encode('utf-8')
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])


def decrypt_record(record):
    """Decrypt a single record."""
    if "canary" not in record:
        print(f"Warning: record has no 'canary' field, skipping decryption: {record.get('id', 'unknown')}")
        return record
    
    key = record["canary"]
    decrypted_record = record.copy()
    
    # Decrypt the prompt field
    if "prompt" in record and record["prompt"]:
        try:
            encrypted_prompt = base64.b64decode(record["prompt"])
            decrypted_record["prompt"] = xor_decrypt(encrypted_prompt, key).decode('utf-8')
        except Exception as e:
            print(f"Warning: failed to decrypt prompt (id: {record.get('id', 'unknown')}): {e}")
    
    # Decrypt the answer field
    if "answer" in record and record["answer"]:
        try:
            encrypted_answer = base64.b64decode(record["answer"])
            decrypted_record["answer"] = xor_decrypt(encrypted_answer, key).decode('utf-8')
        except Exception as e:
            print(f"Warning: failed to decrypt answer (id: {record.get('id', 'unknown')}): {e}")
    
    return decrypted_record


def main():
    parser = argparse.ArgumentParser(description="Download and decrypt xbench/DeepSearch-2510")
    parser.add_argument(
        "--output",
        default=os.path.join("data", "deepsearch", "DeepSearch-2510_decrypted.json"),
        help="Output JSON path (default: data/deepsearch/DeepSearch-2510_decrypted.json)",
    )
    args = parser.parse_args()

    print("Loading dataset xbench/DeepSearch-2510...")
    try:
        ds = load_dataset("xbench/DeepSearch-2510")
        print("Dataset loaded.")
        print(f"Splits: {list(ds.keys())}")
        
        all_decrypted_data = {}
        
        for split_name, split_data in ds.items():
            print(f"\nProcessing split: {split_name}")
            print(f"Records: {len(split_data)}")
            
            decrypted_records = []
            for i, record in enumerate(split_data):
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(split_data)} records...")
                
                decrypted_record = decrypt_record(record)
                decrypted_records.append(decrypted_record)
            
            all_decrypted_data[split_name] = decrypted_records
            print(f"Split {split_name} done.")
        
        output_file = args.output
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        print(f"\nSaving decrypted data to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_decrypted_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved to {output_file}")
        
        print("\n=== Dataset statistics ===")
        for split_name, records in all_decrypted_data.items():
            print(f"{split_name}: {len(records)} records")
            if records:
                print(f"  Fields: {list(records[0].keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
