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
EXCLUDED_REGISTERS = {"fi", "ed", "ra", "LY", "it", "SP", "av"}
TEXT_MIN_LENGTH = 50
THRESHOLD = 0.4
TOKEN_RATIO = 0.75
TARGET_TOKENS = 30_000_000_000
PACKAGES = 160
NUM_PROCESSES = 64  # Number of CPU cores to use


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


def process_file_pair(args):
    """Process a single pair of text and prediction files."""
    (
        file_text,
        file_pred,
        shared_tokens,
        shared_tokens_lock,
        completed_list,
        completed_lock,
        file_locks,
    ) = args
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
                        with shared_tokens_lock:
                            with completed_lock:
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
                                    file_lock = file_locks[label_to_save]

                                    with file_lock:
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
                if i > 0 and i % 10000 == 0:
                    # Update shared counts
                    with shared_tokens_lock:
                        with completed_lock:
                            for label, count in local_updates.items():
                                shared_tokens[label] += count
                                if (
                                    shared_tokens[label] >= TARGET_TOKENS
                                    and label not in completed_list
                                ):
                                    completed_list.append(label)
                            local_updates.clear()

                            # Print progress
                            print("\nToken counts per register:")
                            completed_set = set(completed_list)
                            for register in sorted(get_all_possible_registers()):
                                print(
                                    f"{register}: {shared_tokens[register]}/{TARGET_TOKENS} tokens"
                                )
                            print(
                                f"Completed registers: {len(completed_set)}/{len(get_all_possible_registers())}"
                            )

                            # Check if all registers are complete
                            if len(completed_set) == len(get_all_possible_registers()):
                                return

            # Final update for remaining tokens
            with shared_tokens_lock:
                with completed_lock:
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
    shared_tokens_lock = Lock()  # Separate lock for the shared dictionary
    completed_list = manager.list()  # Using list instead of set
    completed_lock = manager.Lock()  # Lock for the completed list
    file_locks = {register: manager.Lock() for register in get_all_possible_registers()}

    # Initialize the output directory and token counters
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for label in get_all_possible_registers():
        shared_tokens[label] = 0
        os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

    # Generate all file pairs to process
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
                    (
                        file_text,
                        file_pred,
                        shared_tokens,
                        shared_tokens_lock,
                        completed_list,
                        completed_lock,
                        file_locks,
                    )
                )

    # Process files in parallel
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        pool.map(process_file_pair, file_pairs)

    # Print final results
    print("\nSampling complete. Final token counts:")
    for register in sorted(get_all_possible_registers()):
        print(f"{register}: {shared_tokens[register]}/{TARGET_TOKENS} tokens")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
