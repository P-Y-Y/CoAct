#!/usr/bin/env python3
"""
Compute usage statistics for a judge_result JSON file and update it in place.

Usage: python update_judge_stats.py <input_json_file> [output_json_file]

- average input/output tokens over correct samples
- avg_comm_bytes = avg_comm_input_bytes + avg_comm_output_bytes
- global average steps (if missing)
- average run time of correct samples (if missing)
- global per-sample averages for all tools (if missing): LLM, Jina, Serper, Comm, Visual, Text, Audio
"""

import json
import sys
from pathlib import Path


def calculate_correct_sample_stats(data):
    """Token statistics over correct samples."""
    correct_samples = [
        detail for detail in data.get("details", [])
        if detail.get("gpt_result") == "YES"
    ]
    
    if not correct_samples:
        print("Warning: no correct samples found (gpt_result == 'YES')")
        return None
    
    # Token usage
    total_input_tokens = 0
    total_output_tokens = 0
    
    for sample in correct_samples:
        token_usage = sample.get("token_usage", {})
        llm_tokens = token_usage.get("llm", {})
        total_input_tokens += llm_tokens.get("total_input_tokens", 0)
        total_output_tokens += llm_tokens.get("total_output_tokens", 0)
    
    num_correct = len(correct_samples)
    
    stats = {
        "num_correct_samples": num_correct,
        "avg_input_tokens": round(total_input_tokens / num_correct, 2),
        "avg_output_tokens": round(total_output_tokens / num_correct, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens
    }
    
    return stats


def calculate_correct_sample_duration(data):
    """Average run time over correct samples."""
    correct_samples = [
        detail for detail in data.get("details", [])
        if detail.get("gpt_result") == "YES"
    ]
    
    if not correct_samples:
        print("Warning: no correct samples found (gpt_result == 'YES')")
        return None
    
    total_duration = 0
    valid_samples = 0
    
    for sample in correct_samples:
        if "duration_sec" in sample:
            total_duration += sample["duration_sec"]
            valid_samples += 1
    
    if valid_samples == 0:
        print("Warning: no correct samples with duration_sec found")
        return None
    
    avg_duration = round(total_duration / valid_samples, 2)
    
    return {
        "avg_correct_duration_sec": avg_duration,
        "total_correct_duration_sec": total_duration,
        "valid_correct_samples": valid_samples
    }


def calculate_global_comm_stats(data):
    """Global average communication statistics (over all samples)."""
    global_comm = data.get("global_comm_usage", {})
    
    if not global_comm:
        print("Warning: global_comm_usage not found")
        return None
    
    total_input_bytes = global_comm.get("total_input_bytes", 0)
    total_output_bytes = global_comm.get("total_output_bytes", 0)
    total_calls = global_comm.get("total_calls", 0)
    
    # Total number of samples
    total_samples = data.get("total", 0)
    
    if total_samples == 0:
        print("Warning: total is 0, cannot compute averages")
        return None
    
    # Average bytes: (input + output) / number of samples
    total_bytes = total_input_bytes + total_output_bytes
    avg_comm_bytes = round(total_bytes / total_samples, 2)
    avg_comm_calls = round(total_calls / total_samples, 2)
    avg_comm_input_bytes = round(total_input_bytes / total_samples, 2)
    avg_comm_output_bytes = round(total_output_bytes / total_samples, 2)
    
    return {
        "avg_comm_bytes": avg_comm_bytes,
        "avg_comm_calls": avg_comm_calls,
        "avg_comm_input_bytes": avg_comm_input_bytes,
        "avg_comm_output_bytes": avg_comm_output_bytes,
        "total_bytes": total_bytes,
        "total_samples": total_samples
    }


def calculate_all_global_averages(data):
    """Compute global per-sample averages for all tools (over all samples)."""
    total_samples = data.get("total", 0)
    
    if total_samples == 0:
        print("Warning: total is 0, cannot compute averages")
        return None
    
    stats = {}
    
    # LLM statistics
    if "global_llm_usage" in data:
        llm = data["global_llm_usage"]
        # Prefer total_duration_sec, fall back to total_call_duration_sec
        total_call_duration = llm.get("total_duration_sec", 0)
        if total_call_duration == 0:
            total_call_duration = llm.get("total_call_duration_sec", 0)
        
        # If global_llm_usage has no duration field, compute it from details
        if total_call_duration == 0:
            details = data.get("details", [])
            for detail in details:
                token_usage = detail.get("token_usage", {})
                llm_usage = token_usage.get("llm", {})
                total_call_duration += llm_usage.get("total_duration_sec", 0)
            
            if total_call_duration > 0:
                print(f"LLM total_duration_sec computed from details: {total_call_duration}")
        
        stats["avg_llm_calls"] = round(llm.get("total_calls", 0) / total_samples, 2)
        stats["avg_llm_input_tokens"] = round(llm.get("total_input_tokens", 0) / total_samples, 2)
        stats["avg_llm_output_tokens"] = round(llm.get("total_output_tokens", 0) / total_samples, 2)
        stats["avg_llm_tokens"] = round(llm.get("total_tokens", 0) / total_samples, 2)
        stats["avg_llm_duration_sec"] = round(total_call_duration / total_samples, 2)
    else:
        print("Warning: global_llm_usage field missing")
    
    # Jina statistics
    if "global_jina_usage" in data:
        jina = data["global_jina_usage"]
        total_jina_duration = jina.get("total_duration_sec", 0)
        
        # If global_jina_usage has no total_duration_sec, compute it from details
        if total_jina_duration == 0:
            details = data.get("details", [])
            for detail in details:
                token_usage = detail.get("token_usage", {})
                jina_usage = token_usage.get("jina", {})
                total_jina_duration += jina_usage.get("total_duration_sec", 0)
            
            if total_jina_duration > 0:
                print(f"Jina total_duration_sec computed from details: {total_jina_duration}")
        
        stats["avg_jina_calls"] = round(jina.get("total_calls", 0) / total_samples, 2)
        stats["avg_jina_tokens"] = round(jina.get("total_tokens", 0) / total_samples, 2)
        stats["avg_jina_duration_sec"] = round(total_jina_duration / total_samples, 2)
    
    # Serper statistics
    if "global_serper_usage" in data:
        serper = data["global_serper_usage"]
        total_serper_duration = serper.get("total_duration_sec", 0)
        
        # If global_serper_usage has no total_duration_sec, compute it from details
        if total_serper_duration == 0:
            details = data.get("details", [])
            for detail in details:
                token_usage = detail.get("token_usage", {})
                serper_usage = token_usage.get("serper", {})
                total_serper_duration += serper_usage.get("total_duration_sec", 0)
            
            if total_serper_duration > 0:
                print(f"Serper total_duration_sec computed from details: {total_serper_duration}")
        
        stats["avg_serper_calls"] = round(serper.get("total_calls", 0) / total_samples, 2)
        stats["avg_serper_credits"] = round(serper.get("total_credits", 0) / total_samples, 2)
        stats["avg_serper_duration_sec"] = round(total_serper_duration / total_samples, 2)
    
    # Communication statistics
    if "global_comm_usage" in data:
        comm = data["global_comm_usage"]
        total_input_bytes = comm.get("total_input_bytes", 0)
        total_output_bytes = comm.get("total_output_bytes", 0)
        stats["avg_comm_calls"] = round(comm.get("total_calls", 0) / total_samples, 2)
        stats["avg_comm_input_bytes"] = round(total_input_bytes / total_samples, 2)
        stats["avg_comm_output_bytes"] = round(total_output_bytes / total_samples, 2)
        stats["avg_comm_bytes"] = round((total_input_bytes + total_output_bytes) / total_samples, 2)
    
    # Visual statistics (if present)
    if "run_meta" in data and "visual_usage" in data["run_meta"]:
        visual = data["run_meta"]["visual_usage"]
        total_visual_duration = visual.get("total_duration_sec", 0)
        
        # If total_duration_sec is missing, compute it from details
        if total_visual_duration == 0:
            details = data.get("details", [])
            for detail in details:
                token_usage = detail.get("token_usage", {})
                visual_usage = token_usage.get("visual", {})
                total_visual_duration += visual_usage.get("total_duration_sec", 0)
        
        stats["avg_visual_calls"] = round(visual.get("total_calls", 0) / total_samples, 2)
        stats["avg_visual_tokens"] = round(visual.get("total_tokens", 0) / total_samples, 2)
        stats["avg_visual_duration_sec"] = round(total_visual_duration / total_samples, 2)
    
    # Text statistics (if present)
    if "run_meta" in data and "text_usage" in data["run_meta"]:
        text = data["run_meta"]["text_usage"]
        total_text_duration = text.get("total_duration_sec", 0)
        
        # If total_duration_sec is missing, compute it from details
        if total_text_duration == 0:
            details = data.get("details", [])
            for detail in details:
                token_usage = detail.get("token_usage", {})
                text_usage = token_usage.get("text", {})
                total_text_duration += text_usage.get("total_duration_sec", 0)
        
        stats["avg_text_calls"] = round(text.get("total_calls", 0) / total_samples, 2)
        stats["avg_text_tokens"] = round(text.get("total_tokens", 0) / total_samples, 2)
        stats["avg_text_duration_sec"] = round(total_text_duration / total_samples, 2)
    
    # Audio statistics (if present)
    if "run_meta" in data and "audio_usage" in data["run_meta"]:
        audio = data["run_meta"]["audio_usage"]
        total_audio_duration = audio.get("total_duration_sec", 0)
        
        # If total_duration_sec is missing, compute it from details
        if total_audio_duration == 0:
            details = data.get("details", [])
            for detail in details:
                token_usage = detail.get("token_usage", {})
                audio_usage = token_usage.get("audio", {})
                total_audio_duration += audio_usage.get("total_duration_sec", 0)
        
        stats["avg_audio_calls"] = round(audio.get("total_calls", 0) / total_samples, 2)
        stats["avg_audio_tokens"] = round(audio.get("total_tokens", 0) / total_samples, 2)
        stats["avg_audio_duration_sec"] = round(total_audio_duration / total_samples, 2)
    
    return stats


def calculate_global_avg_steps(data):
    """Compute the global average number of steps."""
    details = data.get("details", [])
    
    if not details:
        print("Warning: no sample details found")
        return None
    
    total_steps = 0
    valid_samples = 0
    
    for detail in details:
        if "steps" in detail:
            total_steps += detail["steps"]
            valid_samples += 1
    
    if valid_samples == 0:
        print("Warning: no samples with step information found")
        return None
    
    avg_steps = round(total_steps / valid_samples, 2)
    
    return {
        "avg_steps": avg_steps,
        "total_steps": total_steps,
        "valid_samples": valid_samples
    }


def update_json_stats(input_file, output_file=None):
    """Update the statistics in the JSON file."""
    
    # Read the JSON file
    print(f"Reading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Global per-sample averages for all tools (if missing)
    print("\n=== Global average statistics ===")
    
    # Check whether global averages need to be computed
    need_global_avg_calc = "global_sample_stats" not in data
    
    if need_global_avg_calc:
        print("Global average statistics not found, computing...")
        global_stats = calculate_all_global_averages(data)
        
        if global_stats:
            # Add global average run time
            total_samples = data.get("total", 0)
            if total_samples > 0:
                avg_run_duration = data.get("avg_run_duration_sec", 0)
                if avg_run_duration > 0:
                    global_stats["avg_duration_sec"] = avg_run_duration
                else:
                    # If avg_run_duration_sec is missing, compute it from details
                    details = data.get("details", [])
                    total_duration = sum(d.get("duration_sec", 0) for d in details if "duration_sec" in d)
                    valid_samples = sum(1 for d in details if "duration_sec" in d)
                    if valid_samples > 0:
                        global_stats["avg_duration_sec"] = round(total_duration / valid_samples, 2)
            
            data["global_sample_stats"] = global_stats
            
            print(f"Total samples: {data.get('total')}")
            
            # Print average run time
            if "avg_duration_sec" in global_stats:
                print(f"\nAverage run time: {global_stats.get('avg_duration_sec')} s")
            
            # Print LLM statistics
            if "avg_llm_calls" in global_stats:
                print(f"\nLLM statistics:")
                print(f"  avg calls: {global_stats.get('avg_llm_calls')}")
                print(f"  avg input tokens: {global_stats.get('avg_llm_input_tokens')}")
                print(f"  avg output tokens: {global_stats.get('avg_llm_output_tokens')}")
                print(f"  avg total tokens: {global_stats.get('avg_llm_tokens')}")
                print(f"  Avg time (s): {global_stats.get('avg_llm_duration_sec')}")
            
            # Print Jina statistics
            if "avg_jina_calls" in global_stats:
                print(f"\nJina statistics:")
                print(f"  avg calls: {global_stats.get('avg_jina_calls')}")
                print(f"  avg tokens: {global_stats.get('avg_jina_tokens')}")
                print(f"  Avg time (s): {global_stats.get('avg_jina_duration_sec')}")
            
            # Print Serper statistics
            if "avg_serper_calls" in global_stats:
                print(f"\nSerper statistics:")
                print(f"  avg calls: {global_stats.get('avg_serper_calls')}")
                print(f"  avg credits: {global_stats.get('avg_serper_credits')}")
                print(f"  Avg time (s): {global_stats.get('avg_serper_duration_sec')}")
            
            # Print communication statistics
            if "avg_comm_calls" in global_stats:
                print(f"\nCommunication statistics:")
                print(f"  avg calls: {global_stats.get('avg_comm_calls')}")
                print(f"  avg input bytes: {global_stats.get('avg_comm_input_bytes')}")
                print(f"  avg output bytes: {global_stats.get('avg_comm_output_bytes')}")
                print(f"  avg total bytes: {global_stats.get('avg_comm_bytes')}")
            
            # Print Visual statistics (if present)
            if "avg_visual_calls" in global_stats:
                print(f"\nVisual statistics:")
                print(f"  avg calls: {global_stats.get('avg_visual_calls')}")
                print(f"  avg tokens: {global_stats.get('avg_visual_tokens')}")
                print(f"  Avg time (s): {global_stats.get('avg_visual_duration_sec')}")
            
            # Print Text statistics (if present)
            if "avg_text_calls" in global_stats:
                print(f"\nText statistics:")
                print(f"  avg calls: {global_stats.get('avg_text_calls')}")
                print(f"  avg tokens: {global_stats.get('avg_text_tokens')}")
                print(f"  Avg time (s): {global_stats.get('avg_text_duration_sec')}")
            
            # Print Audio statistics (if present)
            if "avg_audio_calls" in global_stats:
                print(f"\nAudio statistics:")
                print(f"  avg calls: {global_stats.get('avg_audio_calls')}")
                print(f"  avg tokens: {global_stats.get('avg_audio_tokens')}")
                print(f"  Avg time (s): {global_stats.get('avg_audio_duration_sec')}")
    else:
        print("Global average statistics already present, skipping")
        print(f"global_sample_stats has {len(data['global_sample_stats'])} entries")
    
    # 2. Token statistics over correct samples
    print("\n=== Token statistics of correct samples ===")
    
    # Check whether token statistics already exist
    need_token_calc = False
    if "correct_sample_stats" not in data:
        need_token_calc = True
    else:
        token_fields = ["avg_input_tokens", "avg_output_tokens", "total_input_tokens", "total_output_tokens"]
        missing_fields = [f for f in token_fields if f not in data["correct_sample_stats"]]
        if missing_fields:
            need_token_calc = True
            print(f"Missing fields: {missing_fields}")
    
    if need_token_calc:
        print("Computing token statistics of correct samples...")
        correct_stats = calculate_correct_sample_stats(data)
        
        if correct_stats:
            print(f"Correct samples: {correct_stats['num_correct_samples']}")
            print(f"Avg input tokens: {correct_stats['avg_input_tokens']}")
            print(f"Avg output tokens: {correct_stats['avg_output_tokens']}")
            print(f"Total input tokens: {correct_stats['total_input_tokens']}")
            print(f"Total output tokens: {correct_stats['total_output_tokens']}")
            
            # Write into correct_sample_stats
            if "correct_sample_stats" not in data:
                data["correct_sample_stats"] = {}
            
            data["correct_sample_stats"]["avg_input_tokens"] = correct_stats["avg_input_tokens"]
            data["correct_sample_stats"]["avg_output_tokens"] = correct_stats["avg_output_tokens"]
            data["correct_sample_stats"]["total_input_tokens"] = correct_stats["total_input_tokens"]
            data["correct_sample_stats"]["total_output_tokens"] = correct_stats["total_output_tokens"]
    else:
        print("Token statistics already present, skipping")
        print(f"Avg input tokens: {data['correct_sample_stats'].get('avg_input_tokens')}")
        print(f"Avg output tokens: {data['correct_sample_stats'].get('avg_output_tokens')}")
    
    # 3. Average run time of correct samples in correct_sample_stats
    print("\n=== Average run time of correct samples ===")
    if "correct_sample_stats" in data and "avg_duration_sec" not in data["correct_sample_stats"]:
        print("avg_duration_sec not found in correct_sample_stats, computing...")
        
        # Prefer the existing avg_correct_duration_sec
        if "avg_correct_duration_sec" in data:
            data["correct_sample_stats"]["avg_duration_sec"] = data["avg_correct_duration_sec"]
            print(f"Using existing avg_correct_duration_sec: {data['avg_correct_duration_sec']} s")
        else:
            # Otherwise compute it
            duration_stats = calculate_correct_sample_duration(data)
            if duration_stats:
                data["correct_sample_stats"]["avg_duration_sec"] = duration_stats["avg_correct_duration_sec"]
                print(f"Avg run time of correct samples: {duration_stats['avg_correct_duration_sec']} s")
    elif "correct_sample_stats" in data:
        print(f"avg_duration_sec already in correct_sample_stats: {data['correct_sample_stats'].get('avg_duration_sec')} s")
    
    # 4. avg_comm_bytes
    print("\n=== Communication bytes ===")
    if "correct_sample_stats" in data and "avg_comm_bytes" not in data["correct_sample_stats"]:
        print("avg_comm_bytes missing, computing...")
        avg_input_bytes = data["correct_sample_stats"].get("avg_comm_input_bytes", 0)
        avg_output_bytes = data["correct_sample_stats"].get("avg_comm_output_bytes", 0)
        avg_comm_bytes = round(avg_input_bytes + avg_output_bytes, 2)
        
        data["correct_sample_stats"]["avg_comm_bytes"] = avg_comm_bytes
        
        print(f"Avg comm input bytes: {avg_input_bytes}")
        print(f"Avg comm output bytes: {avg_output_bytes}")
        print(f"Avg comm total bytes: {avg_comm_bytes}")
    elif "correct_sample_stats" in data:
        print(f"avg_comm_bytes already present: {data['correct_sample_stats']['avg_comm_bytes']}")
    else:
        print("correct_sample_stats missing, skipping avg_comm_bytes")
    
    # 5. Global average steps (if missing)
    print("\n=== Global average steps ===")
    if "avg_steps" not in data:
        print("Global average steps not found, computing...")
        global_steps_stats = calculate_global_avg_steps(data)
        
        if global_steps_stats:
            data["avg_steps"] = global_steps_stats["avg_steps"]
            print(f"Global avg steps: {global_steps_stats['avg_steps']}")
            print(f"Total steps: {global_steps_stats['total_steps']}")
            print(f"Valid samples: {global_steps_stats['valid_samples']}")
    else:
        print(f"Global avg steps already present: {data['avg_steps']}")
    
    # 6. Root-level average run time of correct samples (if missing)
    print("\n=== Root-level average run time of correct samples ===")
    if "avg_correct_duration_sec" not in data:
        print("avg_correct_duration_sec not found, computing...")
        duration_stats = calculate_correct_sample_duration(data)
        
        if duration_stats:
            data["avg_correct_duration_sec"] = duration_stats["avg_correct_duration_sec"]
            print(f"Avg run time of correct samples: {duration_stats['avg_correct_duration_sec']} s")
            print(f"Total run time of correct samples: {duration_stats['total_correct_duration_sec']} s")
            print(f"Valid correct samples: {duration_stats['valid_correct_samples']}")
    else:
        print(f"avg_correct_duration_sec already present: {data['avg_correct_duration_sec']} s")
    
    # Save the updated JSON
    if output_file is None:
        output_file = input_file
    
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Update done.")
    
    return data


def main():
    """Entry point."""
    if len(sys.argv) < 2:
        print("Usage: python update_judge_stats.py <input_json_file> [output_json_file]")
        print("Example: python update_judge_stats.py input.json")
        print("Example: python update_judge_stats.py input.json output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"Error: file does not exist - {input_file}")
        sys.exit(1)
    
    try:
        update_json_stats(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()