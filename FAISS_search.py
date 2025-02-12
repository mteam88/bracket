import os
import yaml
import markdown
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Constants
EMBEDDINGS_DIR = "embeddings"
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "tweet_index.faiss")
TWEET_DATA_PATH = os.path.join(EMBEDDINGS_DIR, "tweet_data.pkl")
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "tweet_embeddings.npy")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Local model

# Step 1: Read Markdown Files
def load_tweets_from_vault(vault_path):
    tweet_data = []
    
    for filename in os.listdir(vault_path):
        if filename.endswith(".md"):
            with open(os.path.join(vault_path, filename), "r", encoding="utf-8") as f:
                content = f.read()
                parts = content.split("---", 2)  # Split YAML frontmatter
                if len(parts) < 3:
                    continue
                
                metadata = yaml.safe_load(parts[1])  # Parse YAML
                text = parts[2].strip()  # Extract tweet text

                tweet_data.append({"id": metadata.get("tweet_id"), "text": text, "metadata": metadata})

    return tweet_data

# Step 2: Convert Tweets to Embeddings
def embed_tweets(tweets):
    texts = [t["text"] for t in tweets]
    embeddings = model.encode(texts, convert_to_numpy=True)  # Get vectors
    return np.array(embeddings)

# Step 3: Store in FAISS
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # Vector dimension
    index = faiss.IndexFlatL2(d)  # L2 Distance (good for semantic search)
    index.add(embeddings)  # Add vectors
    return index

# Step 4: Search Function
def search_tweets(index, tweets, query, top_k=20):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        if i < len(tweets):
            results.append(tweets[i])
    
    return results

def save_data(index, tweets, embeddings):
    """Save FAISS index, tweet data, and embeddings to disk"""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Save tweet data
    with open(TWEET_DATA_PATH, 'wb') as f:
        pickle.dump(tweets, f)
    
    # Save embeddings
    np.save(EMBEDDINGS_PATH, embeddings)

def load_data():
    """Load FAISS index, tweet data, and embeddings from disk if they exist"""
    if not all(os.path.exists(p) for p in [FAISS_INDEX_PATH, TWEET_DATA_PATH, EMBEDDINGS_PATH]):
        return None, None, None
        
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(TWEET_DATA_PATH, 'rb') as f:
        tweets = pickle.load(f)
        
    embeddings = np.load(EMBEDDINGS_PATH)
    
    return index, tweets, embeddings

def initialize_or_load(vault_path, force_rebuild=False):
    """Initialize the search system, either loading from disk or creating new embeddings"""
    if not force_rebuild:
        index, tweets, embeddings = load_data()
        if all(x is not None for x in [index, tweets, embeddings]):
            print("Loaded existing index and embeddings from disk")
            return index, tweets, embeddings
    
    print("Creating new embeddings and index...")
    tweets = load_tweets_from_vault(vault_path)
    embeddings = embed_tweets(tweets)
    index = create_faiss_index(embeddings)
    
    # Save to disk
    save_data(index, tweets, embeddings)
    print("Saved new index and embeddings to disk")
    
    return index, tweets, embeddings

# Example usage
if __name__ == "__main__":
    vault_path = "Bracket Project Vault/Tweets"
    
    # Initialize or load the system
    index, tweets, embeddings = initialize_or_load(vault_path)
    
    # Search Example
    query = "centralized sequencers overrated and underrated"
    results = search_tweets(index, tweets, query)

    # Show results
    for r in results:
        print(f"Tweet ID: {r['id']}\nText: {r['text']}\n")
