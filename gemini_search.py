import os
import yaml
import markdown
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.llms.gemini import Gemini
import chromadb

# Set up Gemini
os.environ["GOOGLE_API_KEY"] = ""  # Replace with your Gemini API key
llm = Gemini(model="models/gemini-pro")

# Set up global settings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model
Settings.llm = llm  # Use Gemini as the LLM

# Set up paths
OBSIDIAN_VAULT_PATH = "Bracket Project Vault/Tweets"  # Change to your actual path
CHROMA_DB_PATH = "embeddings/chromadb"  # Where embeddings will be stored

# Step 1: Read Markdown Files
def load_markdown_files(vault_path):
    docs = []
    
    print(f"📂 Reading files from: {vault_path}")
    for filename in os.listdir(vault_path):
        if filename.endswith(".md"):
            with open(os.path.join(vault_path, filename), "r", encoding="utf-8") as f:
                content = f.read()
                parts = content.split("---", 2)  # Split YAML frontmatter
                if len(parts) < 3:
                    continue
                
                metadata = yaml.safe_load(parts[1])  # Parse YAML
                text = parts[2].strip()  # Extract tweet text

                # Create a LlamaIndex Document
                docs.append({"id": metadata.get("tweet_id"), "text": text, "metadata": metadata})
    
    print(f"📝 Loaded {len(docs)} tweets")
    if len(docs) > 0:
        print(f"📌 Sample tweet: {docs[0]['text'][:200]}...")
    return docs

# Step 2: Convert Markdown to LlamaIndex Documents
def create_documents(tweets):
    from llama_index.core.schema import Document

    def clean_metadata(metadata):
        # Convert all values to strings to ensure compatibility
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)) or value is None:
                cleaned[key] = value
            else:
                # Convert other types to string representation
                cleaned[key] = str(value)
        return cleaned

    return [
        Document(
            text=tweet["text"],
            metadata=clean_metadata(tweet["metadata"])
        )
        for tweet in tweets
    ]

# Step 3: Set Up Persistent ChromaDB
def setup_chromadb():
    # Create the directory if it doesn't exist
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    # Initialize the persistent client
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="tweets")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    return vector_store

# Step 4: Create or Load an Index
def get_or_create_index(documents):
    vector_store = setup_chromadb()

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if ChromaDB already has embeddings
    try:
        # Try to get the count of items in the collection
        collection = vector_store._collection
        count = collection.count()
        
        if count > 0:
            print("🔄 Loading existing ChromaDB index...")
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        else:
            print("📌 Creating a new ChromaDB index...")
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, storage_context=storage_context)
    except Exception as e:
        print(f"⚠️ Error checking collection, creating new index: {str(e)}")
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, storage_context=storage_context)

    return index

# Step 5: Query the Index
def query_index(index, query, top_k=5):
    # Get response from RAG
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="compact"
    )
    rag_response = query_engine.query(query)
    
    # Construct a prompt that combines RAG results with a request for additional knowledge
    combined_prompt = f"""Based on the following context from a tweet database AND your general knowledge, please provide a comprehensive answer.

Context from tweets: {rag_response.response}

Question: {query}

Please provide a complete answer that combines both the specific information from the tweets AND your general knowledge about the topic."""

    # Get combined response using Gemini
    general_response = llm.complete(combined_prompt)
    
    return general_response.text

# Run the pipeline
tweets = load_markdown_files(OBSIDIAN_VAULT_PATH)
documents = create_documents(tweets)
print(f"\n🔍 Creating/loading index with {len(documents)} documents")
index = get_or_create_index(documents)

# Search Example
query = "what is a based rollup?"
print(f"\n❓ Searching for: {query}")
results = query_index(index, query)

# Display Results
print("\n📊 Combined Knowledge Results:")
print(results)
