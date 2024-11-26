import os
import json
from collections import defaultdict

# Configuration
ROOT_DIR = "/scratch/project_462000353/HPLT-REGISTERS"
LANG = "eng_Latn"
BASE_DIR_TEXT = f"{ROOT_DIR}/splits/deduplicated/{LANG}"
BASE_DIR_PRED = f"{ROOT_DIR}/predictions/deduplicated/{LANG}"
OUTPUT_DIR = f"{ROOT_DIR}/samples-30B-by-register"
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

# Token counters and completed registers
tokens_per_register = defaultdict(int)
completed_registers = set()


def get_all_possible_registers():
    """Get a set of all possible registers (parents and children), excluding the filtered ones."""
    all_registers = set(LABEL_HIERARCHY.keys())
    for children in LABEL_HIERARCHY.values():
        all_registers.update(children)
    return all_registers - EXCLUDED_REGISTERS


# Initialize the output directory and token counters
os.makedirs(OUTPUT_DIR, exist_ok=True)
for label in LABEL_HIERARCHY:
    if label not in EXCLUDED_REGISTERS:
        os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
        for sub in LABEL_HIERARCHY[label]:
            if sub not in EXCLUDED_REGISTERS:
                os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def check_completion_status():
    """Check if all non-excluded registers have reached their target tokens."""
    all_registers = get_all_possible_registers()
    for register in all_registers:
        if tokens_per_register[register] >= TARGET_TOKENS:
            completed_registers.add(register)

    return len(completed_registers) == len(all_registers)


def process_files():
    for dir_num in range(1, PACKAGES + 1):
        dir_path_text = os.path.join(BASE_DIR_TEXT, str(dir_num))
        dir_path_pred = os.path.join(BASE_DIR_PRED, str(dir_num))
        if not os.path.exists(dir_path_text) or not os.path.exists(dir_path_pred):
            continue

        for file_num in range(8):
            file_text = os.path.join(dir_path_text, f"0{file_num}.jsonl")
            file_pred = os.path.join(dir_path_pred, f"0{file_num}.jsonl")
            if not os.path.exists(file_text) or not os.path.exists(file_pred):
                continue

            # Check completion status before processing each file
            if check_completion_status():
                print("Sampling complete. Final token counts:")
                for register in sorted(get_all_possible_registers()):
                    print(
                        f"{register}: {tokens_per_register[register]}/{TARGET_TOKENS} tokens"
                    )
                return

            with open(file_text, "r") as f_text, open(file_pred, "r") as f_pred:
                for i, (line_text, line_pred) in enumerate(zip(f_text, f_pred)):
                    process_line(line_text, line_pred)

                    if i % 10000 == 0:
                        print("\nToken counts per register:")
                        for register in sorted(get_all_possible_registers()):
                            print(
                                f"{register}: {tokens_per_register[register]}/{TARGET_TOKENS} tokens"
                            )
                        print(
                            f"Completed registers: {len(completed_registers)}/{len(get_all_possible_registers())}"
                        )

                        if check_completion_status():
                            print("\nSampling complete. Final token counts:")
                            for register in sorted(get_all_possible_registers()):
                                print(
                                    f"{register}: {tokens_per_register[register]}/{TARGET_TOKENS} tokens"
                                )
                            return


def process_line(line_text, line_pred):
    try:
        data_text = json.loads(line_text)
        data_pred = json.loads(line_pred)

        text = data_text.get("text", "")
        register_probs = data_pred.get("register_probabilities", {})

        assert data_pred["id"] == data_text["id"], "id mismatch"

        # Exclude short text
        if len(text) < TEXT_MIN_LENGTH:
            return

        # Calculate word count and tokens
        word_count = len(text.split())
        tokens = int(word_count / TOKEN_RATIO)

        # Step 1: Binarize probabilities and filter out excluded registers
        active_labels = [
            label
            for label, prob in register_probs.items()
            if prob >= THRESHOLD and label not in EXCLUDED_REGISTERS
        ]

        # Step 2: Check conditions for saving
        if len(active_labels) == 1:
            label = active_labels[0]
            if (
                label not in completed_registers
                and tokens_per_register[label] < TARGET_TOKENS
            ):
                tokens_per_register[label] += tokens
                save_sample(label, text, register_probs)

        elif len(active_labels) == 2:
            # Check if it's a valid parent-child pair
            child = check_parent_child(active_labels)
            if (
                child
                and child not in completed_registers
                and tokens_per_register[child] < TARGET_TOKENS
            ):
                tokens_per_register[child] += tokens
                save_sample(child, text, register_probs)

    except Exception as e:
        print(f"Error processing line: {e}")


def check_parent_child(active_labels):
    """Check if the two active labels form a valid parent-child pair."""
    for parent, children in LABEL_HIERARCHY.items():
        if parent in active_labels:
            for child in children:
                if child in active_labels and child not in EXCLUDED_REGISTERS:
                    return child
    return None


def save_sample(label, text, register_probs):
    """Save the sample to the corresponding register's directory."""
    output_path = os.path.join(OUTPUT_DIR, label, "eng_Latn.jsonl")
    sample = {"text": text, "register_probabilities": register_probs}

    with open(output_path, "a") as f_out:
        f_out.write(json.dumps(sample) + "\n")


# Run the sampling process
process_files()
