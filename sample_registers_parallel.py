import os
import json
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import time
import fcntl

# Configuration
ROOT_DIR = "/scratch/project_462000353/HPLT-REGISTERS"
LANG = "eng_Latn"
BASE_DIR_TEXT = f"{ROOT_DIR}/splits/deduplicated/{LANG}"
BASE_DIR_PRED = f"{ROOT_DIR}/predictions/deduplicated/{LANG}"
OUTPUT_DIR = f"{ROOT_DIR}/samples-30B-by-register-parallel"
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

# Add locks for printing and file writing
print_lock = mp.Lock()
file_locks = {}


def get_all_possible_registers():
    """Get a set of all possible registers (parents and children), excluding the filtered ones."""
    all_registers = set(LABEL_HIERARCHY.keys())
    for children in LABEL_HIERARCHY.values():
        all_registers.update(children)
    return all_registers - EXCLUDED_REGISTERS


def get_file_lock(filepath):
    """Get or create a lock for a specific file."""
    if filepath not in file_locks:
        file_locks[filepath] = mp.Lock()
    return file_locks[filepath]


def check_parent_child(active_labels):
    """Check if the two active labels form a valid parent-child pair."""
    for parent, children in LABEL_HIERARCHY.items():
        if parent in active_labels:
            for child in children:
                if child in active_labels and child not in EXCLUDED_REGISTERS:
                    return child
    return None


def log_progress(dir_num, file_num, i, manager_dict):
    """Thread-safe logging of progress."""
    if i % 10000 == 0:
        with print_lock:
            print(f"\nProgress - Directory {dir_num}, File {file_num}, Line {i}")
            print("\nToken counts per register:")
            for register in sorted(get_all_possible_registers()):
                print(f"{register}: {manager_dict[register]}/{TARGET_TOKENS} tokens")

            # Calculate completed registers
            completed = sum(
                1
                for reg in get_all_possible_registers()
                if manager_dict[reg] >= TARGET_TOKENS
            )
            total = len(get_all_possible_registers())
            print(f"Completed registers: {completed}/{total}")


def save_sample(label, text, register_probs):
    """Save the sample to the corresponding register's directory with proper file locking."""
    output_path = os.path.join(OUTPUT_DIR, label, "eng_Latn.jsonl")
    sample = {"text": text, "register_probabilities": register_probs}

    # Get the lock for this specific file
    file_lock = get_file_lock(output_path)

    # Use the lock to ensure exclusive file access
    with file_lock:
        with open(output_path, "a") as f_out:
            # Use system-level file locking for extra safety
            fcntl.flock(f_out.fileno(), fcntl.LOCK_EX)
            try:
                f_out.write(json.dumps(sample) + "\n")
                f_out.flush()  # Ensure write is completed before releasing lock
                os.fsync(f_out.fileno())  # Force write to disk
            finally:
                fcntl.flock(f_out.fileno(), fcntl.LOCK_UN)


def process_single_directory(dir_num, manager_dict):
    """Process a single directory number."""
    dir_path_text = os.path.join(BASE_DIR_TEXT, str(dir_num))
    dir_path_pred = os.path.join(BASE_DIR_PRED, str(dir_num))
    if not os.path.exists(dir_path_text) or not os.path.exists(dir_path_pred):
        return

    for file_num in range(8):
        file_text = os.path.join(dir_path_text, f"0{file_num}.jsonl")
        file_pred = os.path.join(dir_path_pred, f"0{file_num}.jsonl")
        if not os.path.exists(file_text) or not os.path.exists(file_pred):
            continue

        with open(file_text, "r") as f_text, open(file_pred, "r") as f_pred:
            for i, (line_text, line_pred) in enumerate(zip(f_text, f_pred)):
                try:
                    # Log progress periodically
                    log_progress(dir_num, file_num, i, manager_dict)

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

                    # Check the token count before acquiring any locks
                    if len(active_labels) == 1:
                        label = active_labels[0]
                        with manager_dict.get_lock():
                            if manager_dict[label] < TARGET_TOKENS:
                                manager_dict[label] += tokens
                                # Only save if we actually updated the count
                                save_sample(label, text, register_probs)

                    elif len(active_labels) == 2:
                        child = check_parent_child(active_labels)
                        if child:
                            with manager_dict.get_lock():
                                if manager_dict[child] < TARGET_TOKENS:
                                    manager_dict[child] += tokens
                                    # Only save if we actually updated the count
                                    save_sample(child, text, register_probs)

                except Exception as e:
                    with print_lock:
                        print(f"Error processing line in directory {dir_num}: {e}")


def main():
    start_time = time.time()

    # Initialize the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for label in LABEL_HIERARCHY:
        if label not in EXCLUDED_REGISTERS:
            os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
            for sub in LABEL_HIERARCHY[label]:
                if sub not in EXCLUDED_REGISTERS:
                    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # Create a manager and shared dictionary for token counts
    with mp.Manager() as manager:
        shared_dict = manager.dict()
        for register in get_all_possible_registers():
            shared_dict[register] = 0

        # Initialize file locks
        for register in get_all_possible_registers():
            output_path = os.path.join(OUTPUT_DIR, register, "eng_Latn.jsonl")
            file_locks[output_path] = mp.Lock()

        # Create a pool of workers
        num_cpus = mp.cpu_count()  # or mp.cpu_count() to use all available CPUs
        with mp.Pool(num_cpus) as pool:
            # Process directories in parallel
            process_dir_partial = partial(
                process_single_directory, manager_dict=shared_dict
            )
            pool.map(process_dir_partial, range(1, PACKAGES + 1))

        # Print final results
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print("\nFinal token counts:")
        for register in sorted(get_all_possible_registers()):
            print(f"{register}: {shared_dict[register]}/{TARGET_TOKENS} tokens")


if __name__ == "__main__":
    main()
