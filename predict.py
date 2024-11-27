import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Constants and settings
MAX_LENGTH = 512

# Set up model for speed and precision
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def process_large_file(input_file, chunk_size):
    """Reads a large file in chunks, preserving the original order via indexing."""
    with open(input_file, "r") as f:
        chunk = []
        for idx, line in enumerate(f):
            document = json.loads(line)
            document["original_index"] = idx  # Track original index
            chunk.append(document)
            if len(chunk) >= chunk_size:
                # Sort each chunk by text length before yielding
                chunk.sort(key=lambda x: len(x["text"]), reverse=True)
                yield chunk
                chunk = []
        if chunk:
            # Sort the last chunk if it has any remaining data
            chunk.sort(key=lambda x: len(x["text"]), reverse=True)
            yield chunk


def process_chunk(chunk, batch_size, tokenizer, model, id2label):
    """Process each chunk, predict labels, and save results in original order."""
    results = []
    for i in range(0, len(chunk), batch_size):
        batch = chunk[i : i + batch_size]

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
        probs = torch.sigmoid(logits).cpu().tolist()

        # Store each result with its original index to maintain order
        for id_, idx, prob in zip(ids, original_indices, probs):
            # Create a dictionary of register names and their probabilities
            register_probs = {id2label[str(i)]: round(p, 4) for i, p in enumerate(prob)}

            results.append(
                {
                    "original_index": idx,
                    "id": id_,
                    "register_probabilities": register_probs,
                }
            )

    # Sort results by original index to ensure output order matches input order
    results.sort(key=lambda x: x["original_index"])
    return [
        {
            "id": result["id"],
            "register_probabilities": result["register_probabilities"],
        }
        for result in results
    ]


def main(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    if args.compile:
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=True,
            backend="inductor",
        )
    model.eval()

    # Load id2label mapping from config.json
    # with open(os.path.join(args.model_dir, "config.json"), "r") as f:
    #    config = json.load(f)
    # id2label = config["id2label"]
    id2label = model.config.id2label

    output_file = args.output_file
    total_items = 0
    total_time = 0.0
    with open(output_file, "a") as f:
        for chunk_idx, chunk in enumerate(
            process_large_file(args.input_file, args.chunk_size)
        ):
            start_time = time.perf_counter()

            results = process_chunk(
                chunk,
                args.batch_size,
                tokenizer,
                model,
                id2label,
            )

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

            # Log progress every 100 chunks
            if chunk_idx % 100 == 0:
                print(
                    f"Chunk {chunk_idx}: Throughput = {throughput:.2f} items/s, "
                    f"Average Throughput = {average_throughput:.2f} items/s"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input jsonl file.")
    parser.add_argument("output_file", type=str, help="Path to the output jsonl file.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/xlm-roberta-base",
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="xlm-roberta-base",
        help="Base model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of items per batch.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Number of items per chunk.",
    )
    parser.add_argument(
        "--compile",
        type=int,
        default=1,
        help="Whether to use torch compile.",
    )
    args = parser.parse_args()
    main(args)
