import os
import json
from collections import defaultdict
import multiprocessing as mp
from multiprocessing import Manager, Lock
from itertools import product
import time

# Configuration
ROOT_DIR = "/scratch/project_462000353/HPLT-REGISTERS"
LANG = "eng_Latn"
BASE_DIR_TEXT = f"{ROOT_DIR}/splits/deduplicated/{LANG}"
BASE_DIR_PRED = f"{ROOT_DIR}/predictions/deduplicated/{LANG}"
OUTPUT_DIR = f"{ROOT_DIR}/samples-30B-by-register-p"
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

# Constants
EXCLUDED_REGISTERS = {}
TEXT_MIN_LENGTH = 50
THRESHOLD = 0.4
TOKEN_RATIO = 0.75
TARGET_TOKENS = 30_000_000_000
PACKAGES = 160
NUM_PROCESSES = 128  # Number of CPU cores to use


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


def process_files_chunk(args):
    """Process a chunk of file pairs."""
    file_pairs, shared_tokens, completed_list = args

    # Create process-local locks
    token_update_lock = Lock()
    completion_lock = Lock()
    output_locks = {register: Lock() for register in get_all_possible_registers()}

    for file_text, file_pred in file_pairs:
        process_single_pair(
            file_text,
            file_pred,
            shared_tokens,
            completed_list,
            token_update_lock,
            completion_lock,
            output_locks,
        )


def process_single_pair(
    file_text,
    file_pred,
    shared_tokens,
    completed_list,
    token_update_lock,
    completion_lock,
    output_locks,
):
    """Process a single pair of text and prediction files."""
    local_updates = defaultdict(int)

    try:
        with open(file_text, "r") as f_text, open(file_pred, "r") as f_pred:
            for i, (line_text, line_pred) in enumerate(zip(f_text, f_pred)):
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

                    label_to_save = None
                    if len(active_labels) == 1:
                        label_to_save = active_labels[0]
                    elif len(active_labels) == 2:
                        label_to_save = check_parent_child(active_labels)

                    if label_to_save:
                        # Check if we should process this label
                        with token_update_lock:
                            with completion_lock:
                                completed_set = set(completed_list)
                                if (
                                    label_to_save not in completed_set
                                    and shared_tokens[label_to_save] < TARGET_TOKENS
                                ):
                                    local_updates[label_to_save] += tokens

                                    # Save the sample with minimal locking
                                    output_path = os.path.join(
                                        OUTPUT_DIR, label_to_save, "eng_Latn.jsonl"
                                    )
                                    with output_locks[label_to_save]:
                                        with open(output_path, "a") as f_out:
                                            f_out.write(
                                                json.dumps(
                                                    {
                                                        "text": text,
                                                        "register_probabilities": register_probs,
                                                    }
                                                )
                                                + "\n"
                                            )

                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue

                # Log progress every 10000 lines
                if i > 0 and i % 50000 == 0:
                    # Update shared counts
                    with token_update_lock:
                        with completion_lock:
                            for label, count in local_updates.items():
                                shared_tokens[label] += count
                                if (
                                    shared_tokens[label] >= TARGET_TOKENS
                                    and label not in completed_list
                                ):
                                    completed_list.append(label)
                            local_updates.clear()

                            # Print progress with percentages
                            print("\nToken counts per register:")
                            completed_set = set(completed_list)

                            # Calculate total progress across all registers
                            all_registers = sorted(get_all_possible_registers())
                            total_tokens = sum(
                                shared_tokens[reg] for reg in all_registers
                            )
                            overall_progress = (
                                total_tokens / (TARGET_TOKENS * len(all_registers))
                            ) * 100

                            # Print individual register progress
                            for register in all_registers:
                                percentage = (
                                    shared_tokens[register] / TARGET_TOKENS
                                ) * 100
                                print(f"{register}: {percentage:.2f}%")

                            print(
                                f"\nCompleted registers: {len(completed_set)}/{len(get_all_possible_registers())}"
                            )
                            print(f"Complete progress: {overall_progress:.2f}%")

                            # Check if all registers are complete
                            if len(completed_set) == len(get_all_possible_registers()):
                                return

            # Final update for remaining tokens
            with token_update_lock:
                with completion_lock:
                    for label, count in local_updates.items():
                        shared_tokens[label] += count
                        if (
                            shared_tokens[label] >= TARGET_TOKENS
                            and label not in completed_list
                        ):
                            completed_list.append(label)

    except Exception as e:
        print(f"Error processing file {file_text}: {e}")


def main():
    # Initialize shared state
    manager = Manager()
    shared_tokens = manager.dict()
    completed_list = manager.list()

    # Initialize the output directory and token counters
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for label in get_all_possible_registers():
        shared_tokens[label] = 0
        os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

    # Generate all file pairs
    all_file_pairs = []
    for dir_num in range(1, PACKAGES + 1):
        dir_path_text = os.path.join(BASE_DIR_TEXT, str(dir_num))
        dir_path_pred = os.path.join(BASE_DIR_PRED, str(dir_num))

        if not os.path.exists(dir_path_text) or not os.path.exists(dir_path_pred):
            continue

        for file_num in range(8):
            file_text = os.path.join(dir_path_text, f"0{file_num}.jsonl")
            file_pred = os.path.join(dir_path_pred, f"0{file_num}.jsonl")

            if os.path.exists(file_text) and os.path.exists(file_pred):
                all_file_pairs.append((file_text, file_pred))

    # Split file pairs into chunks for each process
    chunk_size = len(all_file_pairs) // NUM_PROCESSES + 1
    chunks = [
        all_file_pairs[i : i + chunk_size]
        for i in range(0, len(all_file_pairs), chunk_size)
    ]

    # Prepare arguments for each process
    process_args = [(chunk, shared_tokens, completed_list) for chunk in chunks]

    # Process files in parallel
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        pool.map(process_files_chunk, process_args)

    # Print final results
    print("\nSampling complete. Final token counts:")
    for register in sorted(get_all_possible_registers()):
        print(f"{register}: {shared_tokens[register]}/{TARGET_TOKENS} tokens")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
