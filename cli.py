import os
import sys
import argparse
from pathlib import Path
import warnings
import glob
from  src.processor import DocumentLoader
from src.processor import Chunker
from src.processor import Embedder
from src.processor import VectorStore
from src.processor import QueryEngine

from src.processor.intent_detector import IntentDetector
from src.processor.orchestrator import Orchestrator
from src.processor.knowledge.store import KnowledgeStore


from dotenv import load_dotenv
# Suppress deprecation warnings from OpenAI library
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables from .env file
# Ensure the .env file is in the same directory as this script
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

# Init components
# This is where we initialize the intent detector, which will process user inputs and detect intents."""
intent_detector = IntentDetector()

# Initialize knowledge store 
#This is where we load the knowledge store, which contains projects and experiments."""
store = KnowledgeStore()

#This is where we initialize the orchestrator, which will handle the intent processing and interaction with the knowledge store."""
# Initialize orchestrator with the store
orchestrator = Orchestrator(store)


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
def ingest(file_path: str, project:str):

    files = [Path(file).name for file in glob.glob(file_path+"/*")]

    print(files)
    for file_name in files:
        print(f"📥 Ingesting {'/'.join([file_path, file_name])}...")
        docs = doc_loader.load('/'.join([file_path, file_name]))
        chunks = chunker.chunk(docs)
        vectorstore = store_mgr.create_index(chunks)
        store_mgr.save(vectorstore)
        print("✅ File ingested and indexed.")

def query(question: str, project: str = "default_project"):
    print(f"❓ Query: {question}")
    vs_loaded = store_mgr.load(allow_dangerous_deserialization=True)
    query_engine = QueryEngine.QueryEngine(vs_loaded)
    response = query_engine.ask(question)
    print("🤖 Answer:", response)

def intention(question: str):
    
    try:
        result = intent_detector.detect(question)

        if result.get("intent") == "error":
            response = {"status": "error", "message": result["data"]["error"]}
        else:
            response = orchestrator.process_intent(result)
            
    except Exception as e:
        response = {"status": "error", "message": f"Failed to process intent: {str(e)}"}

    print(response)


    


# Command-line routing
# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage:\n  python cli.py ingest <file.pdf>\n  python cli.py query <question>")
#         sys.exit(1)

#     command = sys.argv[1]
#     arg = " ".join(sys.argv[2:])

#     if command == "ingest":
#         ingest(arg)
#     elif command == "query":
#         query(arg)
#     else:
#         print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["intent", "ingest", "query"], help="Action to perform")
    parser.add_argument("input", help="File path, query, or intent text")
    parser.add_argument("--project", help="Project name (default: 'default_project')", default="default_project")

    args = parser.parse_args()

    if args.command == "intent":
        intention(args.input)
    elif args.command == "ingest":
        ingest(args.input, args.project)
    elif args.command == "query":
        query(args.input, args.project)
