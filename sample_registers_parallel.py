import os
import json
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import math
from multiprocessing import Manager
import fcntl
import time
from datetime import datetime

# Configuration
ROOT_DIR = "/scratch/project_462000353/HPLT-REGISTERS"
LANG = "eng_Latn"
BASE_DIR_TEXT = f"{ROOT_DIR}/splits/deduplicated/{LANG}"
BASE_DIR_PRED = f"{ROOT_DIR}/predictions/deduplicated/{LANG}"
OUTPUT_DIR = f"{ROOT_DIR}/samples-30B-by-register-parallelized"
LABEL_HIERARCHY = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

# Separate exclusion list
EXCLUDED_REGISTERS = {"fi", "ed", "ra", "LY", "it", "SP", "av"}
TEXT_MIN_LENGTH = 50
THRESHOLD = 0.4
TOKEN_RATIO = 0.75
TARGET_TOKENS = 30_000_000_000
PACKAGES = 160


def get_all_possible_registers():
    """Get a set of all possible registers (parents and children), excluding the filtered ones."""
    all_registers = set(LABEL_HIERARCHY.keys())
    for children in LABEL_HIERARCHY.values():
        all_registers.update(children)
    return all_registers - EXCLUDED_REGISTERS


def check_parent_child(active_labels):
    """Check if the two active labels form a valid parent-child pair."""
    for parent, children in LABEL_HIERARCHY.items():
        if parent in active_labels:
            for child in children:
                if child in active_labels and child not in EXCLUDED_REGISTERS:
                    return child
    return None


def process_single_file(args):
    """Process a single file pair and return the results."""
    file_text, file_pred, shared_dict = args
    local_results = defaultdict(list)

    try:
        print(f"Starting processing of {os.path.basename(file_text)}")
        start_time = time.time()

        with open(file_text, "r") as f_text, open(file_pred, "r") as f_pred:
            for line_num, (line_text, line_pred) in enumerate(zip(f_text, f_pred)):
                if line_num % 10000 == 0:  # Progress update every 10000 lines
                    print(
                        f"File {os.path.basename(file_text)}: processed {line_num} lines"
                    )

                try:
                    data_text = json.loads(line_text)
                    data_pred = json.loads(line_pred)

                    text = data_text.get("text", "")
                    register_probs = data_pred.get("register_probabilities", {})

                    assert data_pred["id"] == data_text["id"], "id mismatch"

                    if len(text) < TEXT_MIN_LENGTH:
                        continue

                    word_count = len(text.split())
                    tokens = int(word_count / TOKEN_RATIO)

                    active_labels = [
                        label
                        for label, prob in register_probs.items()
                        if prob >= THRESHOLD and label not in EXCLUDED_REGISTERS
                    ]

                    target_label = None
                    if len(active_labels) == 1:
                        target_label = active_labels[0]
                    elif len(active_labels) == 2:
                        target_label = check_parent_child(active_labels)

                    if target_label:
                        # Using the manager's dict for synchronization
                        with shared_dict["lock"]:
                            if (
                                shared_dict["token_counts"][target_label]
                                < TARGET_TOKENS
                            ):
                                shared_dict["token_counts"][target_label] += tokens
                                local_results[target_label].append(
                                    {
                                        "text": text,
                                        "register_probabilities": register_probs,
                                    }
                                )

                except Exception as e:
                    print(f"Error processing line in {file_text}: {e}")

        end_time = time.time()
        with shared_dict["lock"]:
            shared_dict["processed_files"] += 1
            current = shared_dict["processed_files"]

        print(
            f"Completed {os.path.basename(file_text)} ({current}/{shared_dict['total_files']}) in {end_time - start_time:.2f} seconds"
        )

        # Print current token counts every time a file is completed
        with shared_dict["lock"]:
            print("\nCurrent token counts:")
            for register in sorted(shared_dict["token_counts"].keys()):
                print(
                    f"{register}: {shared_dict['token_counts'][register]}/{TARGET_TOKENS} tokens"
                )

    except Exception as e:
        print(f"Error processing file {file_text}: {e}")

    return local_results


def save_results(results, output_dir):
    """Save the accumulated results to files."""
    for register, samples in results.items():
        if not samples:
            continue

        output_path = os.path.join(output_dir, register, "eng_Latn.jsonl")
        os.makedirs(os.path.join(output_dir, register), exist_ok=True)

        print(f"Saving {len(samples)} samples to {output_path}")
        with open(output_path, "a") as f_out:
            fcntl.flock(f_out.fileno(), fcntl.LOCK_EX)
            try:
                for sample in samples:
                    f_out.write(json.dumps(sample) + "\n")
            finally:
                fcntl.flock(f_out.fileno(), fcntl.LOCK_UN)


def main():
    print(f"Starting processing at {datetime.now()}")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_registers = get_all_possible_registers()
    for register in all_registers:
        os.makedirs(os.path.join(OUTPUT_DIR, register), exist_ok=True)

    # Create a manager for shared state
    manager = Manager()
    shared_dict = manager.dict(
        {
            "token_counts": manager.dict(),
            "processed_files": 0,
            "total_files": 0,
            "lock": manager.Lock(),
        }
    )

    for register in all_registers:
        shared_dict["token_counts"][register] = 0

    # Collect all file pairs to process
    file_pairs = []
    for dir_num in range(1, PACKAGES + 1):
        dir_path_text = os.path.join(BASE_DIR_TEXT, str(dir_num))
        dir_path_pred = os.path.join(BASE_DIR_PRED, str(dir_num))

        if not os.path.exists(dir_path_text) or not os.path.exists(dir_path_pred):
            continue

        for file_num in range(8):
            file_text = os.path.join(dir_path_text, f"0{file_num}.jsonl")
            file_pred = os.path.join(dir_path_pred, f"0{file_num}.jsonl")

            if os.path.exists(file_text) and os.path.exists(file_pred):
                file_pairs.append((file_text, file_pred, shared_dict))

    shared_dict["total_files"] = len(file_pairs)

    # Process files in parallel
    num_cpus = min(64, mp.cpu_count())  # Use minimum of 64 or available CPUs
    chunk_size = 1  # Process one file at a time for better progress tracking

    print(f"Processing {len(file_pairs)} file pairs using {num_cpus} CPUs")
    print(f"Chunk size: {chunk_size}")

    with mp.Pool(num_cpus) as pool:
        results_list = pool.map(process_single_file, file_pairs, chunksize=chunk_size)

    print(f"\nAll files processed. Saving final results at {datetime.now()}")

    # Combine and save results
    for partial_results in results_list:
        save_results(partial_results, OUTPUT_DIR)

    print(f"\nProcessing completed at {datetime.now()}")
    print("\nFinal token counts:")
    for register in sorted(all_registers):
        print(
            f"{register}: {shared_dict['token_counts'][register]}/{TARGET_TOKENS} tokens"
        )


if __name__ == "__main__":
    main()
