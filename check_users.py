#!/usr/bin/env python3
"""
Script to check if any of our target users are in the tweet files.
"""

import os
import csv
import json
import random
import argparse
import gc
from collections import defaultdict
import re
import glob
import time

def load_user_ids(data_dir):
    """
    Load user IDs from label.csv and split.csv.
    
    Returns:
        user_ids: Set of user IDs (with 'u' prefix)
    """
    labels_file = os.path.join(data_dir, 'label.csv')
    
    print(f"Loading user IDs from {labels_file}...")
    user_ids = set()
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                user_id_with_prefix = row[0]
                user_ids.add(user_id_with_prefix)
    
    print(f"Loaded {len(user_ids)} user IDs")
    
    # Sample a few user IDs for debugging
    sample_ids = random.sample(list(user_ids), min(5, len(user_ids)))
    print("Sample user IDs:")
    for user_id in sample_ids:
        print(f"  {user_id}")
    
    return user_ids

def check_tweet_file(file_path, user_ids, max_lines=1000):
    """
    Check if any of the user IDs are in the tweet file.
    
    Args:
        file_path: Path to the tweet file
        user_ids: Set of user IDs (with 'u' prefix)
        max_lines: Maximum number of lines to check
    
    Returns:
        found_users: Set of user IDs found in the tweet file
    """
    print(f"Checking {file_path}...")
    found_users = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            
            if i % 100 == 0:
                print(f"  Processed {i} lines...")
            
            try:
                # Try to parse the line as JSON
                tweet = json.loads(line.strip())
                
                # Check if the author_id is in our user_ids
                author_id = tweet.get('author_id')
                if author_id:
                    user_id_with_prefix = f"u{author_id}"
                    if user_id_with_prefix in user_ids:
                        found_users.add(user_id_with_prefix)
                        print(f"  Found user {user_id_with_prefix} in line {i}")
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON
                continue
    
    print(f"Found {len(found_users)} users in {file_path}")
    return found_users

def main():
    parser = argparse.ArgumentParser(description='Check if any of our target users are in the tweet files.')
    parser.add_argument('--data-dir', type=str, default='.', 
                        help='Directory containing the Twibot-22 dataset')
    parser.add_argument('--max-lines', type=int, default=1000,
                        help='Maximum number of lines to check per file')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    max_lines = args.max_lines
    
    # Load user IDs
    user_ids = load_user_ids(data_dir)
    
    # Check tweet files
    tweet_files = sorted(glob.glob(os.path.join(data_dir, 'tweet_*.json')))
    print(f"Found {len(tweet_files)} tweet files")
    
    all_found_users = set()
    for file_path in tweet_files:
        found_users = check_tweet_file(file_path, user_ids, max_lines)
        all_found_users.update(found_users)
    
    print(f"Found {len(all_found_users)} users in all tweet files")
    if all_found_users:
        print("Found users:")
        for user_id in all_found_users:
            print(f"  {user_id}")

if __name__ == "__main__":
    main()
