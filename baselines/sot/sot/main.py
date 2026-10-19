import argparse
import os
import csv
import json
import logging
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import shutil

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
    args, other_args = parser.parse_known_args()

    model_class = get_model_class_from_name(args.model)
    model, other_args = model_class.from_command_line_args(other_args)

    scheduler_class = get_scheduler_class_from_name(args.scheduler)
    scheduler, other_args = scheduler_class.from_command_line_args(
        other_args, model=model
    )
    return args, model, scheduler

def parse_response_text(response_str):
    """
    Parse the response string and split it into think and answer.
    """
    if not response_str:
        return "", ""
    
    response_str = str(response_str).strip()
    
    # Strategy 1: parse the whole string as JSON
    try:
        data = json.loads(response_str)
        if isinstance(data, dict):
            return data.get('think', ""), data.get('answer', "")
    except:
        pass

    # Strategy 2: mixed format <think>...</think>
    think_match = re.search(r'<think>(.*?)</think>', response_str, re.DOTALL | re.IGNORECASE)
    think_content = ""
    answer_content = response_str
    
    if think_match:
        think_content = think_match.group(1).strip()
        answer_content = response_str.replace(think_match.group(0), "").strip()
    
    # Strategy 3: strip markdown json fences
    clean_answer = answer_content
    # Simple cleanup
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
            # Prefer the answer field if it can be parsed
            if json_answer:
                return think_content, json_answer
    except:
        pass

    return think_content, answer_content

# Append results as JSONL (robust to interruptions)
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
    
    # JSONL output path
    json_output_path = os.path.join(args.output_folder, "extracted_results.jsonl")
    # CSV is still written as a backup
    output_data_path = os.path.join(args.output_folder, args.output_data_filename)

    os.makedirs(args.output_folder, exist_ok=True)
    setup_logging(os.path.join(args.output_folder, "log.log"))

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Build skip_requests from the JSONL file (resume support)
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
                        # Each record is assumed to have a "question" field
                        if "question" in record:
                            skip_requests.add(record["question"])
                    except json.JSONDecodeError:
                        continue
            logging.info(f"Found {len(skip_requests)} completed tasks in JSONL. These will be skipped.")
        except Exception as e:
            logging.warning(f"Failed to read existing JSONL file for skipping logic: {e}")
    # ================================================================

    # [Dataset Loading Logic]
    dataset = []
    # Detect the HLE dataset from the data path
    # HLE data structure: { "id":..., "question":..., "image_preview": {"path": "..."} }
    # Assume data-path points to the .json file (e.g. test-00000-of-00001.json)
    if "hle" in args.data_path.lower():
        logging.info("Detected HLE dataset.")
        # HLE dataset handling
        base_dir = os.path.dirname(args.data_path)
        # Determine image directory usually named as {filename}_images
        filename = os.path.basename(args.data_path)
        filename_no_ext = os.path.splitext(filename)[0]
        # Try finding image dir
        image_dir_candidates = [
            os.path.join(base_dir, f"{filename_no_ext}_images"),
            os.path.join(base_dir, "images")
        ]
        image_dir = None
        for cand in image_dir_candidates:
            if os.path.isdir(cand):
                image_dir = cand
                break
        
        if not image_dir:
            logging.warning(f"Could not find HLE image directory. Looked in: {image_dir_candidates}")

        with open(args.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            # Normalize to list
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
            
            for item in raw_data:
                # Construct absolute image path if exists
                image_preview = item.get("image_preview", {})
                image_rel_path = image_preview.get("path") if image_preview else None
                
                final_item = {
                    "question": item.get("question", ""),
                    "id": item.get("id"),
                    "ground_truth": item.get("answer", "")
                }

                if image_rel_path and image_dir:
                    # HLE image path might need just joining
                    final_image_path = os.path.join(image_dir, image_rel_path)
                    if os.path.exists(final_image_path):
                        final_item["file_path"] = final_image_path # standard key for later processing
                    else:
                        logging.warning(f"Image not found at {final_image_path}")
                
                dataset.append(final_item)

        # Configure scheduler for HLE (Visual Only)
        if hasattr(scheduler, "reconfigure_mm_tools"):
            logging.info("Configuring Scheduler for HLE: Enabling only 'visual' tools.")
            scheduler.reconfigure_mm_tools(["visual"])
    else:
        # Standard loading
        with open(args.data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        # [NEW] Check for GAIA to auto-enable MM tools
        if "gaia" in args.data_path.lower():
            if hasattr(scheduler, "reconfigure_mm_tools"):
                logging.info("Configuring Scheduler for GAIA: Enabling all MM tools.")
                # GAIA typically uses all inspector types
                scheduler.reconfigure_mm_tools(["visual", "text", "audio"])


    if args.limit is not None and args.limit > 0:
        logging.info(f"Limiting dataset from {len(dataset)} to {args.limit} items.")
        dataset = dataset[:args.limit]

    # Thread-safe CSV header and lock
    csv_lock = threading.Lock()
    csv_header = None
    
    # GAIA attachments live under $GAIA_DATA_DIR (default: baselines/data/gaia)
    baselines_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gaia_data_dir = os.environ.get("GAIA_DATA_DIR") or os.path.join(baselines_dir, "data", "gaia")

    def process_item(item):
        nonlocal csv_header
        problem = item.get("Question") or item.get("question") or item.get("problem") or item.get("query")
        
        # [Added] Handle GAIA file attachments
        file_name = item.get("file_name")
        gaia_file_path = item.get("file_path")
        
        target_file = None
        # Logic for GAIA or general file path
        if gaia_file_path and os.path.isabs(gaia_file_path):
             target_file = gaia_file_path
        elif gaia_file_path:
             target_file = os.path.join(gaia_data_dir, gaia_file_path)
        elif file_name:
             # Fallback to default validation path if file_path is missing but file_name is present
             target_file = os.path.join(gaia_data_dir, "2023/validation", file_name)
             
        if target_file:
            if os.path.exists(target_file):
                problem += f"\n\n[System Notification]: The attached file to this task. File Path: {target_file}\nPlease use the appropriate inspection tool (e.g., inspect_file_as_image, inspect_file_as_text, inspect_file_as_audio) to read the content of this file if needed to answer the question."
            else:
                logging.warning(f"Attached file not found: {target_file}")

        ground_truth = item.get("answer") or item.get("Answer") or item.get("Final answer") or item.get("ground_truth")

        if problem in skip_requests:
            return

        try:
            # 1. Run inference
            response = scheduler.get_response([problem])
            
            # 2. Add extra fields
            response["ground_truth"] = ground_truth
            response["request"] = problem

            with csv_lock:
                # 3. Save CSV (backup)
                if csv_header is None:
                    csv_header = sorted(response.keys())
                save_csv_row(output_data_path, response, csv_header)

                # 4. Save JSONL (extracted results; primary output used for resuming)
                raw_text = response.get("response", "") or response.get("text", "")
                t_str, a_str = parse_response_text(raw_text)
                
                # Record appended to extracted_results.jsonl
                json_item = {
                    "question": str(response.get("request", "")),
                    "think": t_str,
                    "answer": a_str,
                    "ground_truth": str(response.get("ground_truth", "")),
                    "total_time": str(response.get("total_time", "")),
                    "point_times": str(response.get("point_times", "")),
                    "communication_stats": response.get("communication_stats", {}),
                    "execution_stats": response.get("execution_stats", {}), # per-worker stats from the scheduler
                    "manager_stats": response.get("manager_stats", {})      # manager stats
                }
                append_jsonl_result(json_output_path, json_item)
            
        except Exception as e:
            logging.error(f"Failed to process query: {problem[:50]}... Error: {e}")

    logging.info(f"Starting execution with {args.max_dataset_workers} concurrent dataset workers.")
    
    with ThreadPoolExecutor(max_workers=args.max_dataset_workers) as executor:
        list(tqdm(executor.map(process_item, dataset), total=len(dataset)))

if __name__ == "__main__":
    main()