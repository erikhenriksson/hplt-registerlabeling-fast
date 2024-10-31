import os
import json
import torch
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


def process_chunk(chunk, output_file):
    texts = [item["text"] for item in chunk]
    ids = [item["id"] for item in chunk]
    encodings = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            logits = model(**encodings).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

    results = [{"id": id_, "probs": prob} for id_, prob in zip(ids, probs)]
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
