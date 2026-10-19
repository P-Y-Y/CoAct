#!/usr/bin/env python3
"""Convert Parquet files (e.g. GAIA / HLE snapshots from Hugging Face) to JSON."""
import os
import sys
import json
import argparse
import base64
import io
from glob import glob

try:
    from datasets import Dataset
except Exception as e:
    print("Missing dependency: datasets. Install it with: pip install datasets", file=sys.stderr)
    raise

try:
    from PIL import Image
except ImportError:
    Image = None


def find_parquet_files(root: str) -> list[str]:
    pattern = os.path.join(root, "**", "*.parquet")
    return glob(pattern, recursive=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_safe_name(base_dir: str, file_path: str) -> str:
    rel = os.path.relpath(file_path, base_dir)
    safe = rel.replace(os.sep, "__")
    name = os.path.splitext(safe)[0] + ".json"
    return name


def convert_value(val, image_dir: str = None, image_counter: list = None):
    """
    Recursively convert a value so it is JSON-serializable, handling PIL images.
    - If image_dir is given, images are saved as files and their path is returned.
    - Otherwise images are encoded as base64 strings.
    """
    # PIL images
    if Image is not None and isinstance(val, Image.Image):
        if image_dir and image_counter is not None:
            # Save to file
            ensure_dir(image_dir)
            idx = image_counter[0]
            image_counter[0] += 1
            
            # Determine format
            fmt = val.format or "PNG"
            ext = fmt.lower()
            if ext == "jpeg":
                ext = "jpg"
            
            img_filename = f"image_{idx:04d}.{ext}"
            img_path = os.path.join(image_dir, img_filename)
            val.save(img_path, format=fmt)
            return {"_type": "image", "path": img_filename}
        else:
            # Encode as base64
            buffer = io.BytesIO()
            fmt = val.format or "PNG"
            val.save(buffer, format=fmt)
            b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return {"_type": "image", "format": fmt.lower(), "data": b64_data}
    
    # bytes
    if isinstance(val, bytes):
        return {"_type": "bytes", "data": base64.b64encode(val).decode("utf-8")}
    
    # dict
    if isinstance(val, dict):
        return {k: convert_value(v, image_dir, image_counter) for k, v in val.items()}
    
    # list / tuple
    if isinstance(val, (list, tuple)):
        return [convert_value(v, image_dir, image_counter) for v in val]
    
    # Everything else is returned unchanged
    return val


def write_json_pretty(ds: Dataset, out_path: str, save_images_to_dir: str = None) -> None:
    """
    Write the dataset as pretty-printed JSON.
    If save_images_to_dir is not None, images are saved into that directory.
    """
    rows = list(ds)
    image_counter = [0]
    converted_rows = [convert_value(row, save_images_to_dir, image_counter) for row in rows]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted_rows, f, ensure_ascii=False, indent=2)


def convert_all(input_dir: str, output_dir: str, save_images: bool = True) -> int:
    files = find_parquet_files(input_dir)
    if not files:
        print(f"No Parquet files found in: {input_dir}")
        return 0
    ensure_dir(output_dir)
    converted = 0
    for p in files:
        try:
            ds = Dataset.from_parquet(p)
        except Exception as e:
            print(f"Read failed, skipping: {p} -> {e}", file=sys.stderr)
            continue
        out_name = to_safe_name(input_dir, p)
        out_path = os.path.join(output_dir, out_name)
        
        # Images go into a sibling directory named after the output file
        image_dir = None
        if save_images:
            image_dir_name = os.path.splitext(out_name)[0] + "_images"
            image_dir = os.path.join(output_dir, image_dir_name)
        
        try:
            write_json_pretty(ds, out_path, save_images_to_dir=image_dir)
            converted += 1
            print(f"Converted: {p} -> {out_path}")
            if image_dir and os.path.exists(image_dir):
                print(f"  Images saved to: {image_dir}")
        except Exception as e:
            print(f"Write failed, skipping: {p} -> {e}", file=sys.stderr)
            continue
    return converted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert all Parquet files under a directory (e.g. a Hugging Face dataset snapshot) to JSON")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing Parquet files (e.g. the path returned by download_gaia.py)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for the JSON files (default: ./data)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=True,
        help="Save images as files instead of base64 (default: True)",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_false",
        dest="save_images",
        help="Embed images as base64 instead of saving them as files",
    )
    args = parser.parse_args(argv)

    count = convert_all(args.input_dir, args.output_dir, save_images=args.save_images)
    print(f"Converted {count} Parquet file(s) to JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
