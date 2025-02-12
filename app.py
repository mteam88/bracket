from flask import Flask, render_template, request, jsonify
from faiss_search_lib import TweetSearchEngine
import os

app = Flask(__name__)

# Initialize search engine
VAULT_PATH = "Bracket Project Vault/Tweets"
search_engine = TweetSearchEngine(VAULT_PATH)
search_engine.initialize()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    results = search_engine.search(query)
    print("Backend search results - Tweet IDs:", [{"id": r["id"], "type": type(r["id"])} for r in results])
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True) 