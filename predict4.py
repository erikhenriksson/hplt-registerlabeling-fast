import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
# model = torch.compile(
#    model, mode="reduce-overhead", fullgraph=True, dynamic=True, backend="inductor"
# )
model.eval()


def get_output_filename(input_file):
    dir_path, base_name = os.path.split(input_file)
    return os.path.join(dir_path, base_name.replace(".jsonl", "_register_labels.jsonl"))


def process_large_file(input_file):
    """Reads a large file in chunks, preserving the original order via indexing."""
    with open(input_file, "r") as f:
        chunk = []
        for idx, line in enumerate(f):
            document = json.loads(line)
            document["original_index"] = idx  # Track original index
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


def process_chunk(chunk, output_file):
    """Process each chunk, predict labels, and save results in original order."""
    # List to hold results in the original order
    results = []

    # Split the sorted chunk into smaller batches based on BATCH_SIZE
    for i in range(0, len(chunk), BATCH_SIZE):
        batch = chunk[i : i + BATCH_SIZE]

        # Extract texts, ids, and original indices for the batch
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        original_indices = [item["original_index"] for item in batch]

        # Tokenize with dynamic padding to the longest sequence in the batch
        encodings = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Move the tokenized batch to the device
        encodings = {key: tensor.to(device) for key, tensor in encodings.items()}

        # Run the model on the batch and get logits
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**encodings)
            logits = outputs.logits

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        # Store each result with its original index to maintain order
        for idx, prob in zip(original_indices, probs):
            results.append(
                {
                    "original_index": idx,
                    "id": ids[original_indices.index(idx)],
                    "probs": prob,
                }
            )

    # Sort results by original index to ensure output order matches input order
    results.sort(key=lambda x: x["original_index"])
    return [{"id": result["id"], "probs": result["probs"]} for result in results]


def main(input_file):
    output_file = get_output_filename(input_file)
    with open(output_file, "a") as f:  # Use "a" to append to the file
        for chunk in tqdm(process_large_file(input_file), desc="Processing Chunks"):
            results = process_chunk(chunk)
            # Write results to the output file in the original order
            f.write("\n".join(json.dumps(result) for result in results) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Efficient processing of large datasets."
    )
    parser.add_argument("input_file", type=str, help="Path to the input jsonl file.")
    args = parser.parse_args()
    main(args.input_file)
