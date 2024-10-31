import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from io import BufferedWriter

# Constants and settings
MODEL_DIR = "models/xlm-roberta-base"
BATCH_SIZE = 64
CHUNK_SIZE = 1024
MAX_LENGTH = 512

# Set up model for speed and precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model = torch.compile(
    model, mode="reduce-overhead", fullgraph=True, dynamic=True, backend="inductor"
)
model.eval()


def get_output_filename(input_file):
    dir_path, base_name = os.path.split(input_file)
    return os.path.join(dir_path, base_name.replace(".jsonl", "_register_labels.jsonl"))


def process_large_file(input_file):
    """Reads a large file in chunks, pre-sorting each chunk by text length before yielding."""
    with open(input_file, "r") as f:
        chunk = []
        for line in f:
            document = json.loads(line)
            chunk.append(document)
            if len(chunk) >= CHUNK_SIZE:
                # Sort each chunk by text length before yielding
                chunk.sort(key=lambda x: len(x["text"]), reverse=True)
                yield chunk
                chunk = []
        if chunk:
            # Sort the last chunk if it has any remaining data
            chunk.sort(key=lambda x: len(x["text"]), reverse=True)
            yield chunk


def tokenize_and_sort(chunk):
    """Tokenize documents in a chunk and prepare for batching."""
    texts = [item["text"] for item in chunk]
    ids = [item["id"] for item in chunk]

    # Tokenize texts with padding=False, to keep each sequence length unique
    encodings = tokenizer(texts, padding=False, truncation=True, max_length=MAX_LENGTH)

    # Extract the tokenized input ids and attention masks
    tokenized_data = []
    for i, encoding in enumerate(encodings["input_ids"]):
        tokenized_data.append(
            {
                "id": ids[i],
                "tokens": {key: torch.tensor(encodings[key][i]) for key in encodings},
                "length": len(
                    encoding
                ),  # Length is already sorted due to sorting in process_large_file
                "original_index": i,  # Store original index for reordering
            }
        )

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


def process_chunk(chunk, buffered_writer):
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
    buffered_writer.write("\n".join(json.dumps(result) for result in results) + "\n")


def main(input_file):
    output_file = get_output_filename(input_file)
    with open(output_file, "a") as f:
        buffered_writer = BufferedWriter(f)
        for chunk in tqdm(process_large_file(input_file), desc="Processing Chunks"):
            process_chunk(chunk, buffered_writer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Efficient processing of large datasets."
    )
    parser.add_argument("input_file", type=str, help="Path to the input jsonl file.")
    args = parser.parse_args()
    main(args.input_file)
