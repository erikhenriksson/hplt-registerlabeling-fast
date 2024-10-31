import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np

# Constants and settings
MODEL_DIR = "models/xlm-roberta-base"
BATCH_SIZE = 64  # Consider tuning this based on GPU memory
CHUNK_SIZE = 1024
MAX_LENGTH = 512
NUM_WORKERS = 4  # Adjust based on CPU cores

# Set up model for speed and precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def load_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # Optimize model for inference
    model.to(device)
    model.eval()

    # Optimize transformer attention patterns
    if hasattr(model, "config"):
        model.config.use_cache = True

    # Fuse batch normalization layers
    model = torch.jit.optimize_for_inference(torch.jit.script(model))
    return tokenizer, model


def parallel_tokenize(texts, tokenizer):
    """Tokenize texts in parallel using ThreadPoolExecutor"""

    def _tokenize(text):
        return tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        return list(executor.map(_tokenize, texts))


def process_chunk(chunk, model, tokenizer, output_file):
    """Process each chunk with optimized batch processing"""
    texts = [item["text"] for item in chunk]
    ids = [item["id"] for item in chunk]

    # Parallel tokenization
    encodings = parallel_tokenize(texts, tokenizer)

    # Pre-allocate GPU memory for batches
    max_length = max(len(enc["input_ids"][0]) for enc in encodings)

    results = []

    # Process in optimized batches
    for i in range(0, len(encodings), BATCH_SIZE):
        batch_encodings = encodings[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]

        # Prepare batch tensors
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [enc["input_ids"][0] for enc in batch_encodings],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [enc["attention_mask"][0] for enc in batch_encodings],
            batch_first=True,
            padding_value=0,
        )

        # Move to GPU with memory pinning
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)

        # Run inference with automatic mixed precision
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)

        # Async GPU to CPU transfer
        probs = probs.cpu()

        # Store results
        for j, prob in enumerate(probs):
            results.append({"id": batch_ids[j], "probs": prob.tolist()})

    # Batch write results
    with open(output_file, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def main(input_file):
    # Load model and tokenizer
    tokenizer, model = load_model()

    output_file = input_file.replace(".jsonl", "_register_labels.jsonl")

    # Process chunks with progress bar
    chunks = process_large_file(input_file)
    for chunk in tqdm(chunks, desc="Processing"):
        process_chunk(chunk, model, tokenizer, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input JSONL file")
    args = parser.parse_args()
    main(args.input_file)
