#!/usr/bin/env python3

import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TweetLinkAdder:
    def __init__(self, tweets_dir: str, default_username: str):
        self.tweets_dir = Path(tweets_dir)
        self.default_username = default_username
        self.processed_count = 0
        self.error_count = 0

    def process_file(self, file_path: Path) -> None:
        """Process a single tweet markdown file."""
        try:
            # Read the file content
            content = file_path.read_text(encoding='utf-8')
            
            # Split the content into frontmatter and tweet text
            parts = content.split('---\n', 2)
            if len(parts) != 3:
                logging.error(f"Invalid file format for {file_path}")
                self.error_count += 1
                return
            
            # Parse the YAML frontmatter
            frontmatter = yaml.safe_load(parts[1])
            tweet_text = parts[2]
            
            # Skip if x_link already exists
            if 'x_link' in frontmatter:
                return
            
            # Determine the username for the link
            tweet_id = frontmatter['tweet_id']
            username = self.default_username
            
            # If it's a retweet, we need to handle it differently
            # For now, we'll still use the default username as we don't have retweet source info
            # in the current metadata structure
            
            # Add the X.com link
            frontmatter['x_link'] = f"https://x.com/{username}/status/{tweet_id}"
            
            # Reconstruct the file content
            new_content = "---\n" + yaml.dump(frontmatter, allow_unicode=True) + "---\n" + tweet_text
            
            # Write back to the file
            file_path.write_text(new_content, encoding='utf-8')
            
            self.processed_count += 1
            if self.processed_count % 100 == 0:
                logging.info(f"Processed {self.processed_count} files...")
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            self.error_count += 1

    def process_all_tweets(self) -> None:
        """Process all tweet markdown files in the directory."""
        logging.info(f"Starting to process tweets in {self.tweets_dir}")
        
        for file_path in self.tweets_dir.glob("*.md"):
            self.process_file(file_path)
            
        logging.info(f"Processing complete. Updated {self.processed_count} files.")
        if self.error_count > 0:
            logging.warning(f"Encountered {self.error_count} errors during processing.")

if __name__ == "__main__":
    # Configure these paths as needed
    tweets_dir = "Bracket Project Vault/Tweets"
    default_username = "mteamisloading"
    
    processor = TweetLinkAdder(tweets_dir, default_username)
    processor.process_all_tweets() 