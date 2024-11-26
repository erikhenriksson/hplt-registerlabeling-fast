import os
import json
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import math
from multiprocessing import Manager, Value, Lock
from ctypes import c_longlong

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
    file_text, file_pred, token_counters, register_locks = args
    local_results = defaultdict(list)

    try:
        with open(file_text, "r") as f_text, open(file_pred, "r") as f_pred:
            for line_text, line_pred in zip(f_text, f_pred):
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
                        # Use the lock for this register
                        with register_locks[target_label]:
                            current_count = token_counters[target_label].value
                            if current_count < TARGET_TOKENS:
                                token_counters[target_label].value = (
                                    current_count + tokens
                                )
                                local_results[target_label].append(
                                    {
                                        "text": text,
                                        "register_probabilities": register_probs,
                                    }
                                )

                except Exception as e:
                    print(f"Error processing line in {file_text}: {e}")

    except Exception as e:
        print(f"Error processing file {file_text}: {e}")

    return local_results


def save_results(results, output_dir):
    """Save the accumulated results to files."""
    for register, samples in results.items():
        output_path = os.path.join(output_dir, register, "eng_Latn.jsonl")
        os.makedirs(os.path.join(output_dir, register), exist_ok=True)
        with open(output_path, "a") as f_out:
            for sample in samples:
                f_out.write(json.dumps(sample) + "\n")


def main():
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_registers = get_all_possible_registers()
    for register in all_registers:
        os.makedirs(os.path.join(OUTPUT_DIR, register), exist_ok=True)

    # Create shared token counters and locks for each register
    token_counters = {}
    register_locks = {}
    for register in all_registers:
        token_counters[register] = Value(c_longlong, 0)
        register_locks[register] = Lock()

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
                file_pairs.append(
                    (file_text, file_pred, token_counters, register_locks)
                )

    # Process files in parallel
    num_cpus = 32  # Or mp.cpu_count() for all available CPUs
    chunk_size = max(
        1, len(file_pairs) // (num_cpus * 4)
    )  # Adjust chunk size based on number of CPUs

    print(f"Processing {len(file_pairs)} file pairs using {num_cpus} CPUs")

    with mp.Pool(num_cpus) as pool:
        results_list = pool.map(process_single_file, file_pairs, chunksize=chunk_size)

    # Combine and save results
    for partial_results in results_list:
        save_results(partial_results, OUTPUT_DIR)

    # Print final token counts
    print("\nFinal token counts:")
    for register in sorted(all_registers):
        print(f"{register}: {token_counters[register].value}/{TARGET_TOKENS} tokens")


if __name__ == "__main__":
    main()
