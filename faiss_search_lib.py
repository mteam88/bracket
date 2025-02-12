import os
import yaml
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class TweetSearchEngine:
    def __init__(self, vault_path, embeddings_dir="embeddings"):
        self.vault_path = vault_path
        self.embeddings_dir = embeddings_dir
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Set up paths
        self.faiss_index_path = os.path.join(embeddings_dir, "tweet_index.faiss")
        self.tweet_data_path = os.path.join(embeddings_dir, "tweet_data.pkl")
        self.embeddings_path = os.path.join(embeddings_dir, "tweet_embeddings.npy")
        
        # Initialize
        self.index = None
        self.tweets = None
        self.embeddings = None

    def load_tweets_from_vault(self):
        """Read and parse tweet markdown files from the vault"""
        tweet_data = []
        
        for filename in os.listdir(self.vault_path):
            if filename.endswith(".md"):
                with open(os.path.join(self.vault_path, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    parts = content.split("---", 2)
                    if len(parts) < 3:
                        continue
                    
                    metadata = yaml.safe_load(parts[1])
                    text = parts[2].strip()
                    tweet_data.append({
                        "id": metadata.get("tweet_id"),
                        "text": text,
                        "metadata": metadata
                    })

        return tweet_data

    def embed_tweets(self, tweets):
        """Generate embeddings for tweet texts"""
        texts = [t["text"] for t in tweets]
        return self.model.encode(texts, convert_to_numpy=True)

    def create_faiss_index(self, embeddings):
        """Create and populate FAISS index"""
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index

    def save_data(self):
        """Save search index and data to disk"""
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        faiss.write_index(self.index, self.faiss_index_path)
        
        with open(self.tweet_data_path, 'wb') as f:
            pickle.dump(self.tweets, f)
        
        np.save(self.embeddings_path, self.embeddings)

    def load_data(self):
        """Load search index and data from disk"""
        if not all(os.path.exists(p) for p in [self.faiss_index_path, self.tweet_data_path, self.embeddings_path]):
            return False
            
        self.index = faiss.read_index(self.faiss_index_path)
        
        with open(self.tweet_data_path, 'rb') as f:
            self.tweets = pickle.load(f)
            
        self.embeddings = np.load(self.embeddings_path)
        return True

    def initialize(self, force_rebuild=False):
        """Initialize the search engine"""
        if not force_rebuild and self.load_data():
            return
        
        self.tweets = self.load_tweets_from_vault()
        self.embeddings = self.embed_tweets(self.tweets)
        self.index = self.create_faiss_index(self.embeddings)
        self.save_data()

    def search(self, query, top_k=20):
        """Search tweets by query"""
        if not all([self.index, self.tweets]):
            raise RuntimeError("Search engine not initialized")
            
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.tweets):
                result = self.tweets[i].copy()
                # Convert ID to string to prevent JSON parsing issues with large numbers
                result['id'] = str(result['id'])
                result['score'] = float(1 / (1 + dist))  # Convert distance to similarity score
                result['x_url'] = result['metadata'].get('x_link', '')  # Use x_link from metadata
                results.append(result)
        
        return results 