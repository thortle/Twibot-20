#!/usr/bin/env python3
"""
Ultra-efficient script to extract exactly 200 tweets from Twibot-22 dataset.

This script extracts exactly 100 bot tweets and 100 human tweets from the Twibot-22 dataset,
then stops automatically. It's optimized to use CPU power rather than RAM by:

1. Processing files in small chunks
2. Using multiprocessing with controlled memory usage
3. Implementing aggressive garbage collection
4. Monitoring and reporting memory usage
5. Stopping automatically once the target is reached
"""

import os
import csv
import json
import random
import argparse
import gc
import time
import re
import sys
import multiprocessing as mp
import psutil
from collections import defaultdict

# Constants
CHUNK_SIZE = 1 * 1024 * 1024  # 1MB chunks for processing large files
MAX_MEMORY_PERCENT = 80  # Maximum memory usage percentage before throttling
BATCH_SIZE = 100  # Number of lines to process in a batch

def print_memory_usage():
    """Print current memory usage of the process and system."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024

    # Also get system memory info
    system_memory = psutil.virtual_memory()
    system_percent = system_memory.percent
    system_available_mb = system_memory.available / 1024 / 1024

    print(f"Memory usage: {memory_mb:.2f} MB")
    print(f"System memory: {system_percent:.1f}% used, {system_available_mb:.2f} MB available")

    return system_percent

def force_garbage_collection():
    """Force garbage collection and print memory usage."""
    gc.collect()
    print_memory_usage()

def load_user_metadata(data_dir):
    """
    Load user metadata from label.csv.

    Returns:
        user_to_label: Dictionary mapping user IDs to labels (1 for bot, 0 for human)
        user_id_mapping: Dictionary mapping numeric IDs to IDs with 'u' prefix
    """
    labels_file = os.path.join(data_dir, 'label.csv')

    print(f"Processing labels from {labels_file}...")
    user_to_label = {}
    user_id_mapping = {}  # Map between ID with 'u' prefix and without

    # Process in batches to reduce memory usage
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) >= 2:
                user_id_with_prefix = row[0]
                # Store the label (1 for bot, 0 for human)
                user_to_label[user_id_with_prefix] = 1 if row[1] == 'bot' else 0

                # Create a mapping between IDs with 'u' prefix and without
                if user_id_with_prefix.startswith('u'):
                    user_id_without_prefix = user_id_with_prefix[1:]  # Remove the 'u' prefix
                    user_id_mapping[user_id_without_prefix] = user_id_with_prefix

    # Print some sample mappings for debugging
    print("\nSample user ID mappings (numeric ID -> ID with 'u' prefix):")
    sample_count = 0
    for numeric_id, prefixed_id in list(user_id_mapping.items())[:5]:
        print(f"  {numeric_id} -> {prefixed_id}")
        sample_count += 1

    if sample_count == 0:
        print("  No mappings created! This is a problem.")
    else:
        print(f"  ... and {len(user_id_mapping) - sample_count} more mappings")

    print(f"Loaded {len(user_to_label)} user labels")
    print(f"Created mapping for {len(user_id_mapping)} user IDs")

    return user_to_label, user_id_mapping

def process_json_object(obj, user_id_mapping, user_to_label):
    """
    Process a single JSON object (tweet) to extract text and label.

    Args:
        obj: JSON object representing a tweet
        user_id_mapping: Mapping from numeric IDs to IDs with 'u' prefix
        user_to_label: Dictionary mapping user IDs to labels

    Returns:
        (text, label) tuple if valid, None otherwise
    """
    try:
        # Handle different possible formats of the tweet object
        if isinstance(obj, str):
            # If obj is a string, try to parse it as JSON
            try:
                obj = json.loads(obj)
            except json.JSONDecodeError:
                return None

        if not isinstance(obj, dict):
            return None

        # Extract the author ID
        user_id_raw = obj.get('author_id')
        if user_id_raw is None:
            return None

        # Convert to string (the ID in the tweet file is numeric)
        numeric_user_id = str(user_id_raw)

        # Check if this numeric ID maps to a user with a label
        if numeric_user_id in user_id_mapping:
            # Get the corresponding ID with 'u' prefix
            user_id_with_prefix = user_id_mapping[numeric_user_id]

            # Check if we have a label for this user
            if user_id_with_prefix in user_to_label:
                # Extract the text
                text = obj.get('text', '')
                if text and len(text.strip()) > 10:  # Ensure we have meaningful text
                    # Clean text to reduce memory usage
                    cleaned_text = re.sub(r'https?://\S+', '', text)
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                    # Get the label
                    label = user_to_label[user_id_with_prefix]

                    return (cleaned_text, label)
    except Exception as e:
        # Silently handle any errors and return None
        pass

    return None

def worker_process_chunk(chunk_data, user_id_mapping, user_to_label, bot_count, human_count, bot_target, human_target):
    """
    Worker function to process a chunk of tweet data.

    Args:
        chunk_data: String containing a chunk of the tweet file
        user_id_mapping: Mapping from numeric IDs to IDs with 'u' prefix
        user_to_label: Dictionary mapping user IDs to labels
        bot_count: Current count of bot tweets
        human_count: Current count of human tweets
        bot_target: Target number of bot tweets
        human_target: Target number of human tweets

    Returns:
        Dictionary with 'bot_tweets', 'human_tweets', 'bot_count', 'human_count'
    """
    bot_tweets = []
    human_tweets = []
    local_bot_count = 0
    local_human_count = 0

    # Create a JSON decoder that can handle incomplete objects
    decoder = json.JSONDecoder()

    # Process the chunk
    pos = 0
    tweets_processed = 0

    # Calculate how many more tweets we need
    bot_needed = max(0, bot_target - bot_count)
    human_needed = max(0, human_target - human_count)

    # Stop if we already have enough tweets
    if bot_needed == 0 and human_needed == 0:
        return {
            'bot_tweets': [],
            'human_tweets': [],
            'bot_count': 0,
            'human_count': 0
        }

    # Try to find complete JSON objects in the chunk
    while pos < len(chunk_data):
        try:
            # Skip whitespace
            while pos < len(chunk_data) and chunk_data[pos].isspace():
                pos += 1

            if pos >= len(chunk_data):
                break

            # Try to decode a JSON object
            obj, pos = decoder.raw_decode(chunk_data, pos)
            tweets_processed += 1

            # Process the object
            result = process_json_object(obj, user_id_mapping, user_to_label)
            if result:
                text, label = result

                # Add to appropriate list if we need more of this type
                if label == 1 and local_bot_count < bot_needed:  # Bot
                    bot_tweets.append(text)
                    local_bot_count += 1
                elif label == 0 and local_human_count < human_needed:  # Human
                    human_tweets.append(text)
                    local_human_count += 1

                # Check if we have enough tweets
                if local_bot_count >= bot_needed and local_human_count >= human_needed:
                    break

        except json.JSONDecodeError:
            # If we can't decode a JSON object, move to the next character
            pos += 1
        except Exception as ex:
            # For any other error, skip to the next position
            print(f"Error processing tweet: {ex}")
            pos += 1

        # Log progress for large chunks
        if tweets_processed % 1000 == 0:
            print(f"  Processed {tweets_processed} tweets, found {local_bot_count} bot and {local_human_count} human tweets")

    print(f"Chunk complete: Processed {tweets_processed} tweets, found {local_bot_count} bot and {local_human_count} human tweets")
    return {
        'bot_tweets': bot_tweets,
        'human_tweets': human_tweets,
        'bot_count': local_bot_count,
        'human_count': local_human_count
    }

def process_tweet_file(file_path, user_id_mapping, user_to_label, bot_tweets, human_tweets,
                      bot_target, human_target, num_processes=None):
    """
    Process a tweet file to extract tweets.

    Args:
        file_path: Path to the tweet file
        user_id_mapping: Mapping from numeric IDs to IDs with 'u' prefix
        user_to_label: Dictionary mapping user IDs to labels
        bot_tweets: List to store bot tweets
        human_tweets: List to store human tweets
        bot_target: Target number of bot tweets
        human_target: Target number of human tweets
        num_processes: Number of processes to use

    Returns:
        True if targets are reached, False otherwise
    """
    # Check if we already have enough tweets
    if len(bot_tweets) >= bot_target and len(human_tweets) >= human_target:
        return True

    # Determine number of processes
    if num_processes is None:
        # Use half of available cores to avoid excessive memory usage
        num_processes = max(1, mp.cpu_count() // 2)

    print(f"  Using {num_processes} processes")

    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"  File size: {file_size/1024/1024:.1f} MB")

        # For very large files, use a smaller chunk size
        actual_chunk_size = min(CHUNK_SIZE, max(1024 * 1024, file_size // 100))  # At least 1MB, at most 1/100 of file

        # Create a pool of workers
        pool = mp.Pool(num_processes)

        # Process in chunks to avoid memory issues
        chunks_processed = 0
        offset = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            # Process the file in chunks
            while offset < file_size:
                # Check if we have enough tweets
                if len(bot_tweets) >= bot_target and len(human_tweets) >= human_target:
                    print("  Reached tweet targets, stopping file processing")
                    return True

                # Check memory usage and throttle if necessary
                system_memory_percent = print_memory_usage()
                if system_memory_percent > MAX_MEMORY_PERCENT:
                    print(f"  Memory usage high ({system_memory_percent}%), waiting for GC...")
                    # Wait for garbage collection to free memory
                    time.sleep(5)
                    gc.collect()
                    continue

                # Read a chunk of the file
                f.seek(offset)
                chunk = f.read(actual_chunk_size)

                # Skip if chunk is empty
                if not chunk.strip():
                    offset += len(chunk)
                    continue

                chunks_processed += 1
                if chunks_processed % 5 == 0:
                    print(f"    Processing chunk {chunks_processed}, "
                          f"{offset/file_size*100:.1f}% of file...")
                    print(f"    Current counts: {len(bot_tweets)}/{bot_target} bot tweets, "
                          f"{len(human_tweets)}/{human_target} human tweets")

                # Process chunk in parallel
                chunk_results = pool.apply(worker_process_chunk,
                                         (chunk, user_id_mapping, user_to_label,
                                          len(bot_tweets), len(human_tweets),
                                          bot_target, human_target))

                # Update the results
                bot_tweets.extend(chunk_results['bot_tweets'])
                human_tweets.extend(chunk_results['human_tweets'])

                # Ensure we don't exceed our targets
                if len(bot_tweets) > bot_target:
                    bot_tweets = bot_tweets[:bot_target]
                if len(human_tweets) > human_target:
                    human_tweets = human_tweets[:human_target]

                # Check if we've reached our targets
                if len(bot_tweets) >= bot_target and len(human_tweets) >= human_target:
                    print("  Reached tweet targets, stopping file processing")
                    return True

                # Move to the next chunk
                offset += len(chunk)

                # Force garbage collection between chunks
                gc.collect()

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up pool
        pool.close()
        pool.join()

    # Return whether we've reached our targets
    return len(bot_tweets) >= bot_target and len(human_tweets) >= human_target

def save_dataset_to_files(bot_tweets, human_tweets, output_dir):
    """
    Save the dataset to text files.

    Args:
        bot_tweets: List of bot tweets
        human_tweets: List of human tweets
        output_dir: Directory to save the output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save tweets
    tweets_file = os.path.join(output_dir, 'tweets.txt')
    labels_file = os.path.join(output_dir, 'labels.txt')

    with open(tweets_file, 'w', encoding='utf-8') as f_tweets, \
         open(labels_file, 'w', encoding='utf-8') as f_labels:

        # Write bot tweets
        for tweet in bot_tweets:
            f_tweets.write(f"{tweet}\n")
            f_labels.write("1\n")  # 1 for bot

        # Write human tweets
        for tweet in human_tweets:
            f_tweets.write(f"{tweet}\n")
            f_labels.write("0\n")  # 0 for human

    print(f"Saved dataset to {output_dir}")
    print(f"  Tweets file: {tweets_file}")
    print(f"  Labels file: {labels_file}")

    # Also save as CSV for easier inspection
    csv_file = os.path.join(output_dir, 'dataset.csv')
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("text,label\n")

        # Write bot tweets
        for tweet in bot_tweets:
            # Escape quotes and newlines for CSV
            escaped_text = tweet.replace('"', '""').replace('\n', ' ')
            f.write(f'"{escaped_text}",1\n')

        # Write human tweets
        for tweet in human_tweets:
            # Escape quotes and newlines for CSV
            escaped_text = tweet.replace('"', '""').replace('\n', ' ')
            f.write(f'"{escaped_text}",0\n')

    print(f"  CSV file: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract exactly 200 tweets (100 bot, 100 human) from Twibot-22 dataset')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Directory containing the Twibot-22 dataset')
    parser.add_argument('--output-dir', type=str, default='./extracted_tweets',
                        help='Directory to save the extracted tweets')
    parser.add_argument('--bot-tweets', type=int, default=100,
                        help='Number of bot tweets to extract')
    parser.add_argument('--human-tweets', type=int, default=100,
                        help='Number of human tweets to extract')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes to use (default: half of available cores)')
    args = parser.parse_args()

    start_time = time.time()
    print_memory_usage()

    data_dir = args.data_dir
    output_dir = args.output_dir
    bot_target = args.bot_tweets
    human_target = args.human_tweets
    num_processes = args.processes

    print(f"Starting tweet extraction...")
    print(f"Target: {bot_target} bot tweets, {human_target} human tweets")

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/3] Loading user metadata...")
    user_to_label, user_id_mapping = load_user_metadata(data_dir)

    # Lists to store the tweets
    bot_tweets = []
    human_tweets = []

    print(f"\n[2/3] Processing tweet files to find {bot_target} bot and {human_target} human tweets...")

    # Process all tweet files
    tweet_files = sorted([f for f in os.listdir(data_dir) if f.startswith('tweet_') and f.endswith('.json')])
    print(f"Found {len(tweet_files)} tweet files")

    for i, file_name in enumerate(tweet_files):
        file_path = os.path.join(data_dir, file_name)
        print(f"\nProcessing {file_name} ({i+1}/{len(tweet_files)})...")

        # Process the file
        targets_reached = process_tweet_file(
            file_path,
            user_id_mapping,
            user_to_label,
            bot_tweets,
            human_tweets,
            bot_target,
            human_target,
            num_processes=num_processes
        )

        print(f"  Progress: {len(bot_tweets)}/{bot_target} bot tweets, {len(human_tweets)}/{human_target} human tweets")

        # Check if we've reached our targets
        if targets_reached:
            print("  All tweet targets reached, stopping early")
            break

    print("\n[3/3] Saving extracted tweets...")
    save_dataset_to_files(bot_tweets, human_tweets, output_dir)

    # Print statistics
    print("\nDataset statistics:")
    print(f"  Bot tweets: {len(bot_tweets)}/{bot_target}")
    print(f"  Human tweets: {len(human_tweets)}/{human_target}")
    print(f"  Total tweets: {len(bot_tweets) + len(human_tweets)}/{bot_target + human_target}")

    # Calculate and print execution time
    elapsed_time = time.time() - start_time
    print(f"\nDone! Total execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print_memory_usage()

if __name__ == "__main__":
    main()
