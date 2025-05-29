import os
import sys
from  src.processor import DocumentLoader
from src.processor import Chunker
from src.processor import Embedder
from src.processor import VectorStore
from src.processor import QueryEngine
from dotenv import load_dotenv
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)


groq_api_key = os.getenv("GROQ_API_KEY")  # Or paste directly (not recommended)
openAI = os.getenv("OPENAI_API_KEY")  # Or paste directly (not recommended)

openai_base_url = "https://api.openai.com/v1/chat/completions"


# Setup
doc_loader = DocumentLoader.DocumentLoader()
chunker = Chunker.TextChunker(chunk_size=500, chunk_overlap=50)
embedder = Embedder.Embedder(openAI)
store_mgr = VectorStore.VectorStoreManager(embedder.model)

# # Ingest a file
# docs = doc_loader.load("test.pdf")
# chunks = chunker.chunk(docs)
# vectorstore = store_mgr.create_index(chunks)
# store_mgr.save(vectorstore)

# # Later: Ask a question
# vs_loaded = store_mgr.load(allow_dangerous_deserialization=True)
# query_engine = QueryEngine.QueryEngine(vs_loaded)
# response = query_engine.ask("What is the importance of elliptical shapes in the polarizer model ?")
# print(response)

# CLI interface
def ingest(file_path: str):
    print(f"📥 Ingesting {file_path}...")
    docs = doc_loader.load(file_path)
    chunks = chunker.chunk(docs)
    vectorstore = store_mgr.create_index(chunks)
    store_mgr.save(vectorstore)
    print("✅ File ingested and indexed.")

def query(question: str):
    print(f"❓ Query: {question}")
    vs_loaded = store_mgr.load(allow_dangerous_deserialization=True)
    query_engine = QueryEngine.QueryEngine(vs_loaded)
    response = query_engine.ask(question)
    print("🤖 Answer:", response)

# Command-line routing
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:\n  python cli.py ingest <file.pdf>\n  python cli.py query <question>")
        sys.exit(1)

    command = sys.argv[1]
    arg = " ".join(sys.argv[2:])

    if command == "ingest":
        ingest(arg)
    elif command == "query":
        query(arg)
    else:
        print(f"❌ Unknown command: {command}")

