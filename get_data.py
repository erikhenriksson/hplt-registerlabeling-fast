import os
import sys
import requests
import hashlib
import zstandard as zstd
import io
from pathlib import Path

DATA_FOLDER = "data_deduplicated"
VERIFIED_FILE = "verified_checksums.txt"

# Ensure the directory exists
os.makedirs(DATA_FOLDER, exist_ok=True)


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping download.")
        return True

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    else:
        print(f"Failed to download {url}")
        return False


def is_verified(file_name):
    if os.path.exists(VERIFIED_FILE):
        with open(VERIFIED_FILE, "r") as f:
            verified_files = f.read().splitlines()
        return file_name in verified_files
    return False


def mark_as_verified(file_name):
    with open(VERIFIED_FILE, "a") as f:
        f.write(file_name + "\n")


def verify_checksum(file_path, md5_url):
    file_name = os.path.basename(file_path)
    if is_verified(file_name):
        print(f"Checksum already verified for {file_name}. Skipping.")
        return True

    md5_file = file_path + ".md5"
    if not download_file(md5_url, md5_file):
        print(f"Failed to download checksum for {file_path}")
        return False

    with open(md5_file, "r") as f:
        expected_md5 = f.read().split()[0]

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    actual_md5 = hash_md5.hexdigest()
    if actual_md5 == expected_md5:
        mark_as_verified(file_name)
        return True
    else:
        return False


def shard_and_count_lines(file_path, max_lines_per_shard=1000000):
    # First pass to count the total number of lines
    total_lines = 0
    with open(file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            total_lines = sum(1 for _ in text_stream)

    # Determine the required number of shards
    num_shards = (total_lines + max_lines_per_shard - 1) // max_lines_per_shard
    shard_paths = [f"{file_path}_part{i}.jsonl" for i in range(1, num_shards + 1)]
    shard_files = [
        open(shard_path, "w", encoding="utf-8") for shard_path in shard_paths
    ]
    line_counts = [0] * num_shards
    total_lines = 0

    # Second pass to read, count, and shard lines
    with open(file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            line_number = 0

            for line in text_stream:
                shard_index = min(line_number // max_lines_per_shard, num_shards - 1)
                shard_files[shard_index].write(line)
                line_counts[shard_index] += 1
                line_number += 1

    # Close all shard files and output line counts
    for i, shard_file in enumerate(shard_files):
        shard_file.close()
        print(f"{shard_paths[i]} contains {line_counts[i]} lines.")

    os.remove(file_path)
    print(f"Total lines in {file_path}: {total_lines}")


def process_datasets(lang_code):
    lang_folder = os.path.join(DATA_FOLDER, lang_code)
    os.makedirs(lang_folder, exist_ok=True)
    url_list = f"https://data.hplt-project.org/two/deduplicated/{lang_code}_map.txt"
    response = requests.get(url_list)
    if response.status_code != 200:
        print("Failed to download the URL list.")
        return

    urls = response.text.strip().splitlines()
    for url in urls:
        file_name = os.path.basename(url)
        file_path = os.path.join(lang_folder, file_name)

        # Download the dataset file if not already present
        if download_file(url, file_path):
            print(f"Downloaded {file_name}")

            # Download and verify checksum
            md5_url = url + ".md5"
            if verify_checksum(file_path, md5_url):
                print(f"Checksum verified for {file_name}")

                # Shard file and count lines
                shard_and_count_lines(file_path)
            else:
                print(f"Checksum verification failed for {file_name}")
        else:
            print(f"Skipping {file_name} due to download failure.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_data.py <lang_code>")
        sys.exit(1)

    lang_code = sys.argv[1]
    process_datasets(lang_code)
