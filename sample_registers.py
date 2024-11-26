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
THRESHOLD = 0.4
TOKEN_RATIO = 0.75
# TARGET_TOKENS = 30_000_000_000
TARGET_TOKENS = 500
PACKAGES = 160

# Token counters and completed registers
tokens_per_register = defaultdict(int)
completed_registers = set()

# Prepare output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
for label in LABEL_HIERARCHY:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
    for sub in LABEL_HIERARCHY[label]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def process_files():
    for dir_num in range(1, PACKAGES + 1):
        dir_path_text = os.path.join(BASE_DIR_TEXT, str(dir_num))
        dir_path_pred = os.path.join(BASE_DIR_PRED, str(dir_num))
        if not os.path.exists(dir_path_text) or not os.path.exists(dir_path_pred):
            continue

        for file_num in range(8):
            print(tokens_per_register)
            file_text = os.path.join(dir_path_text, f"0{file_num}.jsonl")
            file_pred = os.path.join(dir_path_pred, f"0{file_num}.jsonl")
            if not os.path.exists(file_text) or not os.path.exists(file_pred):
                continue

            # Check if sampling is complete for all registers
            if len(completed_registers) == len(LABEL_HIERARCHY) + sum(
                len(v) for v in LABEL_HIERARCHY.values()
            ):
                print("Sampling complete")
                return  # Exit early if all sampling is complete

            with open(file_text, "r") as f_text, open(file_pred, "r") as f_pred:
                for line_text, line_pred in zip(f_text, f_pred):
                    process_line(line_text, line_pred)

                    # Stop processing if all registers are done
                    if len(completed_registers) == len(LABEL_HIERARCHY) + sum(
                        len(v) for v in LABEL_HIERARCHY.values()
                    ):
                        return


def process_line(line_text, line_pred):
    try:
        data_text = json.loads(line_text)
        data_pred = json.loads(line_pred)

        text = data_text.get("text", "")
        register_probs = data_pred.get("register_probabilities", {})

        assert data_pred["id"] == data_text["id"], "id mismatch"

        # Exclude short text
        if len(text) < 10:
            return

        # Calculate word count and tokens
        word_count = len(text.split())
        tokens = int(word_count / TOKEN_RATIO)

        # Step 1: Binarize probabilities
        active_labels = [
            label for label, prob in register_probs.items() if prob >= THRESHOLD
        ]

        # Step 2: Check conditions for saving
        if len(active_labels) == 1:
            # Single active label - save for this register
            save_sample(active_labels[0], text, register_probs)
            tokens_per_register[active_labels[0]] += tokens
            if tokens_per_register[active_labels[0]] >= TARGET_TOKENS:
                completed_registers.add(active_labels[0])

        elif len(active_labels) == 2:
            # Check if it's a valid parent-child pair
            parent_child = check_parent_child(active_labels)
            if parent_child:
                save_sample(parent_child, text, register_probs)
                tokens_per_register[parent_child] += tokens
                if tokens_per_register[parent_child] >= TARGET_TOKENS:
                    completed_registers.add(parent_child)

    except Exception as e:
        print(f"Error processing line: {e}")


def check_parent_child(active_labels):
    """Check if the two active labels form a valid parent-child pair."""
    for parent, children in LABEL_HIERARCHY.items():
        if parent in active_labels:
            for child in children:
                if child in active_labels:
                    # Return the child, since we prioritize saving for it
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
