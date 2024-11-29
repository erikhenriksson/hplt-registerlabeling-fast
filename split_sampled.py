import sys
from pathlib import Path
import math
import argparse
from tqdm import tqdm


def create_output_directory(output_path):
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_file_sizes(input_files):
    """Get total size of all input files."""
    return sum(Path(f).stat().st_size for f in input_files)


def count_total_lines(input_files):
    """Count total lines across all input files."""
    total = 0
    total_size = get_file_sizes(input_files)

    # Create progress bar based on file sizes
    with tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Counting lines"
    ) as pbar:
        for file in input_files:
            file_size = Path(file).stat().st_size
            with open(file, "r", encoding="utf-8") as f:
                for _ in f:
                    total += 1
            pbar.update(file_size)

    return total


def split_files(input_files, output_dir, total_lines, num_output_files=8):
    """Split input files into specified number of output files."""
    lines_per_file = math.ceil(total_lines / num_output_files)

    # Create output file handles
    output_files = [
        open(output_dir / f"{i:02d}.jsonl", "w", encoding="utf-8", newline="")
        for i in range(num_output_files)
    ]

    try:
        current_count = 0
        current_file_idx = 0

        # Create progress bar for splitting
        total_size = get_file_sizes(input_files)
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Splitting files"
        ) as pbar:
            # Process each input file
            for input_file in input_files:
                file_size = Path(input_file).stat().st_size
                with open(input_file, "r", encoding="utf-8", newline="") as f:
                    for line in f:
                        # Write to current output file
                        output_files[current_file_idx].write(line)
                        current_count += 1

                        # Move to next file if we've reached the target lines per file
                        if (
                            current_count >= lines_per_file
                            and current_file_idx < num_output_files - 1
                        ):
                            current_count = 0
                            current_file_idx += 1

                pbar.update(file_size)

    finally:
        # Ensure all files are closed properly
        for f in output_files:
            f.close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split JSONL files into equal parts")
    parser.add_argument("input_path", help="Directory containing input JSONL files")
    parser.add_argument("output_path", help="Directory for output files")
    args = parser.parse_args()

    # Convert to Path objects
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Ensure input path exists
    if not input_path.exists():
        print(f"Input directory {input_path} does not exist!")
        return

    # Create output directory if it doesn't exist
    create_output_directory(output_path)

    # Get all input jsonl files
    input_files = sorted(input_path.glob("*.jsonl"))
    input_files = [str(f) for f in input_files]

    if not input_files:
        print(f"No JSONL files found in {input_path}!")
        return

    print(f"Found {len(input_files)} input files")

    total_lines = count_total_lines(input_files)
    print(f"Total lines: {total_lines:,}")

    split_files(input_files, output_path, total_lines)
    print(
        f"Done! Files have been split into {output_path}/00.jsonl through {output_path}/07.jsonl"
    )


if __name__ == "__main__":
    main()
