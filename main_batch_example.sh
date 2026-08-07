#!/bin/bash

# Path to the text file containing image filenames (one per line)
img_filenames_txt="./img_filenames_example.txt"

# Count the number of lines (i.e., images) in the file
line_count=$(wc -l < "$img_filenames_txt")

# Set up LaMa environment variables
pushd ./lama > /dev/null
export TORCH_HOME="$(pwd)"
export PYTHONPATH="$(pwd)"
popd > /dev/null

# Process images in batches of 5
for line_num in $(seq 0 5 $line_count); do
    echo "Processing batch starting from line: $line_num"

    # Run the main amodal completion pipeline
    python main.py \
        --input_dir /your/path/here/ \          # Directory containing input images
        --img_filenames_txt "$img_filenames_txt" \                        # List of image filenames to process
        --json_label_path ./example_annotation.json \  # JSON annotations for labels
        --output_dir ./output_path \                                     # Output directory
        --line_num "$line_num"                                            # Starting line index in filename list

    # Note: Uncomment and customize below if you want to override prompt manually for all images
    # --text_query "What is the TV remote control in this image" \
    # --inpaint_prompt "a TV remote control"

    # Get the process ID and wait for it to finish
    pid=$!
    wait $pid

    # Kill the process explicitly (to avoid memory leakage if needed)
    kill -9 $pid
    echo "Process $pid killed."
done
