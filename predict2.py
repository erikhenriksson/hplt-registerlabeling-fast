import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

# Constants and settings
MODEL_DIR = "models/xlm-roberta-base"
BATCH_SIZE = 64
CHUNK_SIZE = 1000
MAX_LENGTH = 512

# Set up model for speed and precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def get_output_filename(input_file):
    dir_path, base_name = os.path.split(input_file)
    return os.path.join(dir_path, base_name.replace(".jsonl", "_register_labels.jsonl"))


def process_large_file(input_file):
    with open(input_file, "r") as f:
        chunk = []
        for line in f:
            document = json.loads(line)
            chunk.append(document)
            if len(chunk) >= CHUNK_SIZE:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def tokenize_and_sort(chunk):
    """Tokenize documents in a chunk, calculate lengths, and sort by length."""
    texts = [item["text"] for item in chunk]
    ids = [item["id"] for item in chunk]

    # Tokenize texts with padding=False, to keep each sequence length unique
    encodings = tokenizer(texts, padding=False, truncation=True, max_length=MAX_LENGTH)

    # Extract the tokenized input ids and attention masks, and calculate lengths
    tokenized_data = []
    for i, encoding in enumerate(encodings["input_ids"]):
        tokenized_data.append(
            {
                "id": ids[i],
                "tokens": {key: torch.tensor(encodings[key][i]) for key in encodings},
                "length": len(encoding),  # Calculate sequence length for sorting
                "original_index": i,  # Store original index for reordering
            }
        )

    # Sort by length for efficient batching
    tokenized_data.sort(key=lambda x: x["length"], reverse=True)
    return tokenized_data


def collate_batch(batch):
    """Custom collate function to batch data with similar lengths."""
    max_length = max(item["length"] for item in batch)

    # Pad each tensor in the batch to the max length within the batch
    batch_tokens = {
        key: torch.stack(
            [
                F.pad(
                    item["tokens"][key],
                    (0, max_length - item["tokens"][key].shape[0]),
                    value=tokenizer.pad_token_id,
                )
                for item in batch
            ]
        )
        for key in batch[0]["tokens"]
    }

    # Collect ids and original indices for tracking original order
    ids = [item["id"] for item in batch]
    original_indices = [item["original_index"] for item in batch]
    return batch_tokens, ids, original_indices


def process_chunk(chunk, output_file):
    """Process each chunk, predict labels, and save results in original order."""
    sorted_data = tokenize_and_sort(chunk)

    # Pre-allocate results list to restore original order
    results = [None] * len(sorted_data)

    # Prepare DataLoader with length-based sorting for efficient padding
    data_loader = DataLoader(
        sorted_data,
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch,
        shuffle=False,
        pin_memory=True,
    )

    # Process each batch
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_tokens, ids, original_indices in data_loader:
            # Move each tensor in batch_tokens to the device
            batch_tokens = {
                key: tensor.to(device) for key, tensor in batch_tokens.items()
            }

            # Run the model on the batch and get logits
            with torch.no_grad():
                outputs = model(**batch_tokens)
                logits = outputs.logits

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

            # Store results in the correct original index
            for idx, (original_index, prob) in enumerate(zip(original_indices, probs)):
                results[original_index] = {"id": ids[idx], "probs": prob}

    # Write results to output file in the original order
    with open(output_file, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def main(input_file):
    output_file = get_output_filename(input_file)
    for chunk in tqdm(process_large_file(input_file), desc="Processing Chunks"):
        process_chunk(chunk, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Efficient processing of large datasets."
    )
    parser.add_argument("input_file", type=str, help="Path to the input jsonl file.")
    args = parser.parse_args()
    main(args.input_file)
