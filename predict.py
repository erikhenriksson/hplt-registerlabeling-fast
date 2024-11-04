import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

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

# Load id2label mapping from config.json
with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)
id2label = config["id2label"]


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


def process_chunk(chunk):
    """Process each chunk, predict labels, and save results in original order."""
    results = []
    for i in range(0, len(chunk), BATCH_SIZE):
        batch = chunk[i : i + BATCH_SIZE]

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

        # Compute the length of each tokenized text (number of tokens before padding)
        input_lengths = encodings["attention_mask"].sum(dim=1).cpu().tolist()

        # Run the model on the batch and get logits
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**encodings)
            logits = outputs.logits

        # Convert logits to probabilities
        probs = torch.sigmoid(logits).cpu().tolist()

        # Store each result with its original index to maintain order
        for id_, idx, prob, input_length in zip(
            ids, original_indices, probs, input_lengths
        ):
            # Create a dictionary of register names and their probabilities
            register_probs = {id2label[str(i)]: round(p, 4) for i, p in enumerate(prob)}

            # If the text is shorter than 10 tokens, no registers are assigned
            if input_length < 10:
                registers = []
            else:
                # Determine which registers have a probability >= 0.4
                registers = [
                    register for register, p in register_probs.items() if p >= 0.4
                ]

            results.append(
                {
                    "original_index": idx,
                    "id": id_,
                    "registers": registers,
                    "register_probabilities": register_probs,
                }
            )

    # Sort results by original index to ensure output order matches input order
    results.sort(key=lambda x: x["original_index"])
    return [
        {
            "id": result["id"],
            "registers": result["registers"],
            "register_probabilities": result["register_probabilities"],
        }
        for result in results
    ]


def main(input_file):
    output_file = get_output_filename(input_file)
    total_items = 0
    total_time = 0.0
    with open(output_file, "a") as f:
        for chunk_idx, chunk in enumerate(process_large_file(input_file)):
            start_time = time.perf_counter()

            results = process_chunk(chunk)

            f.write("\n".join(json.dumps(result) for result in results) + "\n")

            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            throughput = len(chunk) / elapsed_time if elapsed_time > 0 else float("inf")

            # Update totals
            total_items += len(chunk)
            total_time += elapsed_time
            average_throughput = (
                total_items / total_time if total_time > 0 else float("inf")
            )

            # Log progress every 100 chunks instead of on every iteration
            if chunk_idx % 100 == 0:
                print(
                    f"Chunk {chunk_idx}: Throughput = {throughput:.2f} items/s, "
                    f"Average Throughput = {average_throughput:.2f} items/s"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input jsonl file.")
    args = parser.parse_args()
    main(args.input_file)
