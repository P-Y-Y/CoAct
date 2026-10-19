import argparse
import os
import csv
import json
import logging
import threading
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import shutil

# Make sure we can import modules from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model_class_from_name
from schedulers import get_scheduler_class_from_name
from utils.logging import setup_logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--output-data-filename", type=str, default="data.csv")
    parser.add_argument("--max-dataset-workers", type=int, default=1, help="Number of parallel queries to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of items to process (for debugging)")
    
    # We allow extra args that will be passed to model/scheduler
    args, other_args = parser.parse_known_args()

    # Ensure enable_mm_tools is passed if not present (we need it for HLE)
    if "--enable-mm-tools" not in other_args:
        other_args.append("--enable-mm-tools")

    model_class = get_model_class_from_name(args.model)
    model, other_args = model_class.from_command_line_args(other_args)

    scheduler_class = get_scheduler_class_from_name(args.scheduler)
    scheduler, other_args = scheduler_class.from_command_line_args(
        other_args, model=model
    )
    return args, model, scheduler

def parse_response_text(response_str):
    """
    Parse response string to separate think and answer.
    """
    if not response_str:
        return "", ""
    
    response_str = str(response_str).strip()
    
    # Strategy 1: Try full JSON parse
    try:
        data = json.loads(response_str)
        if isinstance(data, dict):
            return data.get('think', ""), data.get('answer', "")
    except:
        pass

    # Strategy 2: Mixed format <think>...</think>
    think_match = re.search(r'<think>(.*?)</think>', response_str, re.DOTALL | re.IGNORECASE)
    think_content = ""
    answer_content = response_str
    
    if think_match:
        think_content = think_match.group(1).strip()
        answer_content = response_str.replace(think_match.group(0), "").strip()
    
    # Strategy 3: Remove markdown code blocks
    clean_answer = answer_content
    if clean_answer.startswith("```json"):
        clean_answer = clean_answer[7:]
    elif clean_answer.startswith("```"):
        clean_answer = clean_answer[3:]
    if clean_answer.endswith("```"):
        clean_answer = clean_answer[:-3]
    clean_answer = clean_answer.strip()

    try:
        data = json.loads(clean_answer)
        if isinstance(data, dict):
            json_answer = data.get('answer', "")
            if json_answer:
                return think_content, json_answer
    except:
        pass

    return think_content, answer_content

def append_jsonl_result(file_path, result_dict):
    """
    Append a single JSON object as a new line.
    """
    try:
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error appending to JSONL: {e}")

def save_csv_row(file_path, data_dict, header_keys=None):
    file_exists = os.path.isfile(file_path)
    if header_keys is None:
        header_keys = sorted(data_dict.keys())
        
    with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header_keys)
        if not file_exists:
            writer.writeheader()
        
        row = {}
        for k, v in data_dict.items():
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        writer.writerow(row)
    return header_keys

def main():
    args, model, scheduler = parse_args()
    
    # Filter tools: Only keep visual tool
    logging.info("Filtering MM tools to only include 'inspect_file_as_image'...")
    if hasattr(scheduler, "tool_metadata"):
        keys_to_remove = ["inspect_file_as_text", "inspect_file_as_audio"]
        for k in keys_to_remove:
            if k in scheduler.tool_metadata:
                del scheduler.tool_metadata[k]
                logging.info(f"Removed tool: {k}")
    

    json_output_path = os.path.join(args.output_folder, "extracted_results.jsonl")
    output_data_path = os.path.join(args.output_folder, args.output_data_filename)

    os.makedirs(args.output_folder, exist_ok=True)
    setup_logging(os.path.join(args.output_folder, "log.log"))

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Resume logic
    skip_requests = set()
    if os.path.exists(json_output_path):
        try:
            logging.info(f"Checking existing progress in {json_output_path}...")
            with open(json_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        record = json.loads(line)
                        if "question" in record:
                            skip_requests.add(record["question"])
                    except json.JSONDecodeError:
                        continue
            logging.info(f"Found {len(skip_requests)} completed tasks in JSONL. These will be skipped.")
        except Exception as e:
            logging.warning(f"Failed to read existing JSONL file for skipping logic: {e}")

    # Load dataset
    logging.info(f"Loading dataset from {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Verify HLE format briefly
    if len(dataset) > 0:
        first_item = dataset[0]
        logging.info(f"First item keys: {list(first_item.keys())}")
        if "image_preview" in first_item:
            logging.info(f"Sample image_preview: {first_item['image_preview']}")
        else:
            logging.warning("No 'image_preview' found in first item. Is this HLE dataset?")

    if args.limit is not None and args.limit > 0:
        logging.info(f"Limiting dataset from {len(dataset)} to {args.limit} items.")
        dataset = dataset[:args.limit]

    # Deduce images directory
    # If data path is .../data/hle/data/test-00000-of-00001.json
    # Images are likely in .../data/hle/data/test-00000-of-00001_images/
    base_data_path = os.path.splitext(args.data_path)[0]
    images_dir = base_data_path + "_images"
    if not os.path.isdir(images_dir):
        logging.warning(f"Expected images directory not found at: {images_dir}")

    csv_lock = threading.Lock()
    csv_header = None
    
    def process_item(item):
        nonlocal csv_header
        problem = item.get("Question") or item.get("question") or item.get("problem") or item.get("query")
        
        # Handle HLE image
        image_preview = item.get("image_preview")
        
        if image_preview and "path" in image_preview:
            img_filename = image_preview["path"]
            img_abs_path = os.path.join(images_dir, img_filename)
            
            if os.path.exists(img_abs_path):
                 problem += f"\n\n[System Notification]: The attached image to this task is at: {img_abs_path}\nPlease use the 'inspect_file_as_image' tool to view this file if needed to answer the question."
            else:
                 logging.warning(f"Image file not found: {img_abs_path}")
        
        # Original GAIA handling kept for compatibility if mixed, but mainly using HLE logic above
        # ... (omitted for brevity unless needed)

        ground_truth = item.get("answer") or item.get("Answer") or item.get("Final answer") or item.get("ground_truth")

        if problem in skip_requests:
            return

        try:
            # 1. Execute Inference
            response = scheduler.get_response([problem])
            
            # 2. Add extra fields
            response["ground_truth"] = ground_truth
            response["request"] = problem
            
            # 3. Parse final answer
            raw_response = response.get("response", "")
            think_text, answer_text = parse_response_text(raw_response)
            response["parsed_think"] = think_text
            response["parsed_answer"] = answer_text
            
            # 4. Save results
            append_jsonl_result(json_output_path, response)

            # CSV backup
            with csv_lock:
                # Flat tools stats for CSV
                flattened = response.copy()
                if "manager_stats" in flattened and isinstance(flattened["manager_stats"], dict):
                    mm_stats = flattened["manager_stats"].get("mm_tools", {})
                    if mm_stats:
                        flattened["mm_tools_stats"] = json.dumps(mm_stats, ensure_ascii=False)
                
                # Cleanup complex fields
                for k in ["intermediate_steps", "manager_stats", "worker_stats"]:
                    if k in flattened:
                        flattened[k] = json.dumps(flattened[k], ensure_ascii=False)

                csv_header = save_csv_row(output_data_path, flattened, csv_header)
                
        except Exception as e:
            logging.error(f"Error processing item: {problem[:50]}... Error: {e}")

    # Use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_dataset_workers) as executor:
        list(tqdm(executor.map(process_item, dataset), total=len(dataset), desc="Processing items"))

if __name__ == "__main__":
    main()
