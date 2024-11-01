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
model.eval()

# Load id2label mapping from config.json
with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)
id2label = config["id2label"]


def get_output_filename(input_file):
    dir_path, base_name = os.path.split(input_file)
    return os.path.join(dir_path, base_name.replace(".jsonl", "_register_labels.jsonl"))


def process_large_file(input_file):
    with open(input_file, "r") as f:
        chunk = []
        for idx, line in enumerate(f):
            document = json.loads(line)
            document["original_index"] = idx
            chunk.append(document)
            if len(chunk) >= CHUNK_SIZE:
                chunk.sort(key=lambda x: len(x["text"]), reverse=True)
                yield chunk
                chunk = []
        if chunk:
            chunk.sort(key=lambda x: len(x["text"]), reverse=True)
            yield chunk


def process_chunk(chunk):
    results = [None] * len(chunk)  # Pre-allocate list with original order

    for i in range(0, len(chunk), BATCH_SIZE):
        batch = chunk[i : i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        original_indices = [item["original_index"] for item in batch]

        encodings = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {key: tensor.to(device) for key, tensor in encodings.items()}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**encodings)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        for idx, prob, id in zip(original_indices, probs, ids):
            formatted_probabilities = [round(p, 4) for p in prob]
            labels = [id2label[str(i)] for i in range(len(prob))]

            # Place each result in its original index in results list
            results[idx] = {
                "id": id,
                "registers": labels,
                "register_probabilities": formatted_probabilities,
            }

    return results


def main(input_file):
    output_file = get_output_filename(input_file)
    total_items = 0
    total_time = 0.0
    with open(output_file, "a") as f:
        with tqdm(process_large_file(input_file), desc="Processing Chunks") as pbar:
            for chunk in pbar:
                start_time = time.perf_counter()

                results = process_chunk(chunk)

                f.write("\n".join(json.dumps(result) for result in results) + "\n")

                end_time = time.perf_counter()

                elapsed_time = end_time - start_time
                throughput = (
                    len(chunk) / elapsed_time if elapsed_time > 0 else float("inf")
                )

                total_items += len(chunk)
                total_time += elapsed_time
                average_throughput = (
                    total_items / total_time if total_time > 0 else float("inf")
                )

                pbar.set_postfix(
                    {
                        "Throughput (items/s)": f"{throughput:.2f}",
                        "Avg Throughput (items/s)": f"{average_throughput:.2f}",
                    }
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Efficient processing of large datasets."
    )
    parser.add_argument("input_file", type=str, help="Path to the input jsonl file.")
    args = parser.parse_args()
    main(args.input_file)
