#!/usr/bin/env python3

import json
import os
from datetime import datetime
import re
from pathlib import Path
from typing import Dict, List, Any
import logging


X_USERNAME = "mteamisloading"

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class TweetProcessor:
    def __init__(self, tweets_file: str, note_tweets_file: str, output_dir: str):
        self.tweets_file = tweets_file
        self.note_tweets_file = note_tweets_file
        self.output_dir = Path(output_dir)
        self.tweets_dir = self.output_dir / "Tweets"
        self.processed_count = 0
        self.skipped_count = 0
        self.note_tweets = {}  # Will store note tweets by timestamp
        
        # Ensure tweets directory exists
        self.tweets_dir.mkdir(parents=True, exist_ok=True)
        
    def load_note_tweets(self):
        """Load and parse the note tweets file."""
        logging.info(f"Loading note tweets from {self.note_tweets_file}")
        try:
            with open(self.note_tweets_file, 'r', encoding='utf-8') as f:
                content = f.read()
                json_str = content.replace('window.YTD.note_tweet.part0 = ', '')
                note_tweets_data = json.loads(json_str)
                
            # Store note tweets by their timestamp
            for note_tweet in note_tweets_data:
                note_data = note_tweet["noteTweet"]
                # Parse ISO timestamp to datetime
                created_at = datetime.fromisoformat(note_data["createdAt"].replace('Z', '+00:00'))
                self.note_tweets[created_at] = note_data["core"]["text"]
                
            logging.info(f"Loaded {len(self.note_tweets)} note tweets")
            # Log first few timestamps for debugging
            logging.info(f"First few note tweet timestamps: {list(self.note_tweets.keys())[:5]}")
        except Exception as e:
            logging.error(f"Error loading note tweets: {str(e)}")

    def extract_urls(self, tweet_text: str) -> List[str]:
        """Extract URLs from tweet text using regex."""
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+)'
        return re.findall(url_pattern, tweet_text)

    def extract_media_urls(self, tweet: Dict[str, Any]) -> List[str]:
        """Extract media URLs from tweet data."""
        media_urls = []
        if "extended_entities" in tweet and "media" in tweet["extended_entities"]:
            for media in tweet["extended_entities"]["media"]:
                if "media_url_https" in media:
                    media_urls.append(media["media_url_https"])
                elif "video_info" in media and "variants" in media["video_info"]:
                    # Get the highest quality video URL
                    video_variants = media["video_info"]["variants"]
                    if video_variants:
                        media_urls.append(video_variants[0]["url"])
        return media_urls

    def create_markdown_content(self, tweet: Dict[str, Any]) -> str:
        """Create markdown content with YAML frontmatter from tweet data."""
        # Convert tweet timestamp to datetime
        tweet_time = datetime.strptime(
            tweet["created_at"], 
            "%a %b %d %H:%M:%S %z %Y"
        )
        
        # Create YAML frontmatter
        yaml_content = [
            "---",
            f"tweet_id: {tweet['id_str']}",
            f"timestamp: {tweet_time.isoformat()}",
            f"source: {tweet['source']}",
            f"lang: {tweet['lang']}",
            f"favorite_count: {tweet.get('favorite_count', 0)}",
            f"retweet_count: {tweet.get('retweet_count', 0)}",
            f"is_retweet: {tweet.get('retweeted', False)}",
            f"is_favorited: {tweet.get('favorited', False)}",
        ]

        # Add display text range if available
        if "display_text_range" in tweet:
            yaml_content.append(f"display_text_range: {json.dumps(tweet['display_text_range'])}")

        # Add truncated status
        yaml_content.append(f"truncated: {tweet.get('truncated', False)}")
        
        # Handle entities
        entities = tweet.get("entities", {})
        
        # Hashtags
        hashtags = [tag["text"] for tag in entities.get("hashtags", [])]
        if hashtags:
            yaml_content.append(f"hashtags: {json.dumps(hashtags)}")
        
        # Symbols (cashtags)
        symbols = [symbol["text"] for symbol in entities.get("symbols", [])]
        if symbols:
            yaml_content.append(f"symbols: {json.dumps(symbols)}")
        
        # User mentions
        mentions = [{
            "screen_name": mention["screen_name"],
            "name": mention["name"],
            "id": mention["id_str"]
        } for mention in entities.get("user_mentions", [])]
        if mentions:
            yaml_content.append(f"user_mentions: {json.dumps(mentions)}")
        
        # URLs
        urls = [url["expanded_url"] if "expanded_url" in url else url["url"] 
               for url in entities.get("urls", [])]
        if urls:
            yaml_content.append(f"urls: {json.dumps(urls)}")
        
        # Media
        media_urls = self.extract_media_urls(tweet)
        if media_urls:
            yaml_content.append(f"media: {json.dumps(media_urls)}")
        
        # Reply information
        if tweet.get("in_reply_to_status_id_str"):
            reply_info = {
                "reply_to_tweet_id": tweet.get("in_reply_to_status_id_str"),
                "reply_to_user_id": tweet.get("in_reply_to_user_id_str", ""),
                "reply_to_screen_name": tweet.get("in_reply_to_screen_name", "")
            }
            yaml_content.append(f"reply_info: {json.dumps(reply_info)}")
        
        # Quote tweet information
        if tweet.get("is_quote_status"):
            yaml_content.append(f"is_quote_status: true")
            if "quoted_status_id_str" in tweet:
                yaml_content.append(f"quoted_tweet_id: {tweet['quoted_status_id_str']}")
        
        # Location information if available
        if "coordinates" in tweet and tweet["coordinates"]:
            yaml_content.append(f"coordinates: {json.dumps(tweet['coordinates'])}")
        if "place" in tweet and tweet["place"]:
            yaml_content.append(f"place: {json.dumps(tweet['place'])}")
        if "geo" in tweet and tweet["geo"]:
            yaml_content.append(f"geo: {json.dumps(tweet['geo'])}")
            
        # Add x.com link
        # todo handle retweets
        yaml_content.append(f"x_link: https://x.com/{X_USERNAME}/status/{tweet['id_str']}")
            
        yaml_content.append("---\n")
        
        # Add tweet text - check for note tweet version by timestamp
        content = "\n".join(yaml_content)

        content += tweet["full_text"]
        
        # Look for a matching note tweet within a small time window (0.1 second)
        note_tweet_text = None
        for note_time, note_text in self.note_tweets.items():
            if abs((note_time - tweet_time).total_seconds()) <= 0.1:
                note_tweet_text = note_text
                logging.info(f"Found matching note tweet for timestamp {tweet_time.isoformat()}")
                break
                
        if note_tweet_text:
            index = content.find(note_tweet_text[:10])
            if index != -1:
                content = content[:index] + note_tweet_text

            logging.info(f"Using note tweet version for tweet at {tweet_time.isoformat()}")
            logging.info(f"Note tweet text: {note_tweet_text[:100]}...")  # Log first 100 chars of note tweet
        
        return content

    def get_file_path(self, tweet: Dict[str, Any]) -> Path:
        """Generate the file path for a tweet using its ID."""
        tweet_id = tweet["id_str"]
        return self.tweets_dir / f"{tweet_id}.md"

    def process_tweets(self):
        """Process the tweets.js file and create markdown files."""
        logging.info(f"Processing tweets from {self.tweets_file}")
        
        # Load note tweets first
        self.load_note_tweets()
        
        try:
            with open(self.tweets_file, 'r', encoding='utf-8') as f:
                # Skip the "window.YTD.tweets.part0 = " part
                content = f.read()
                json_str = content.replace('window.YTD.tweets.part0 = ', '')
                tweets_data = json.loads(json_str)
                
            for tweet_wrapper in tweets_data:
                tweet = tweet_wrapper["tweet"]
                file_path = self.get_file_path(tweet)
                
                # Create markdown content and save to file
                content = self.create_markdown_content(tweet)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.processed_count += 1
                if self.processed_count % 100 == 0:
                    logging.info(f"Processed {self.processed_count} tweets...")
                    
            logging.info(f"Processing complete. Created {self.processed_count} files.")
            
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Configure these paths as needed
    tweets_file = "input/tweets.js"
    note_tweets_file = "input/note-tweet.js"
    output_dir = "Bracket Project Vault"
    
    processor = TweetProcessor(tweets_file, note_tweets_file, output_dir)
    processor.process_tweets() 