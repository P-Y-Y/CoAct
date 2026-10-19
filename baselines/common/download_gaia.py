"""Download the GAIA benchmark from Hugging Face and load the 2023 level-1 validation split.

GAIA is gated: accept its terms on huggingface.co and log in (`huggingface-cli login`
or set HF_TOKEN) before running. The snapshot path printed below can be passed to
`convert_parquet_to_json.py --input-dir`.
"""
from huggingface_hub import snapshot_download
from datasets import load_dataset

# Download the dataset snapshot into the local Hugging Face cache
data_dir = snapshot_download(
    repo_id="gaia-benchmark/GAIA",
    repo_type="dataset"
)
print(data_dir)

# Load one split
dataset = load_dataset(data_dir, "2023_level1", split="validation")
print(dataset)
