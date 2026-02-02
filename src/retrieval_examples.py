# Part 1, Question 7

import os
from pathlib import Path

import chromadb
from openai import OpenAI

# -----------------------
# Paths & constants
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "ai_agents_jobs_2026"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5

# -----------------------
# Load OpenAI API key
# -----------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = (PROJECT_ROOT / "openai.txt").read_text().strip()

oai = OpenAI(api_key=api_key)

# -----------------------
# Connect to ChromaDB
# -----------------------
print("DB_DIR:", DB_DIR)
print("DB_DIR exists:", DB_DIR.exists())

client = chromadb.PersistentClient(path=str(DB_DIR))
print("Collections:", [c.name for c in client.list_collections()])

collection = client.get_collection(name=COLLECTION_NAME)
print("Using collection:", collection.name, "count:", collection.count())

# -----------------------
# Example queries (Q7)
# -----------------------
queries = [
    "reinforcement learning agent roles in finance",
    "multi-agent orchestration using LangGraph",
    "tool-calling agents in production systems",
    "AI agent engineer job requirements",
    "autonomous agents for enterprise workflows",
]

# -----------------------
# Embed queries
# -----------------------
print("\nEmbedding queries...")
q_resp = oai.embeddings.create(model=EMBED_MODEL, input=queries)
q_vecs = [d.embedding for d in q_resp.data]
print("Query embeddings:", len(q_vecs), "dim:", len(q_vecs[0]))

# -----------------------
# Query ChromaDB
# -----------------------
print("\nQuerying ChromaDB...")
results = collection.query(
    query_embeddings=q_vecs,
    n_results=TOP_K,
    include=["documents", "metadatas", "distances"]
)

# -----------------------
# Print results
# -----------------------
for qi, q in enumerate(queries):
    print("\n" + "=" * 70)
    print(f"Query #{qi+1}: {q}")

    returned = len(results["ids"][qi])
    for rank in range(min(TOP_K, returned)):
        md = results["metadatas"][qi][rank]
        doc = results["documents"][qi][rank]
        dist = results["distances"][qi][rank]

        print(f"\n  Rank {rank+1} | distance={dist:.4f}: {md.get('title','')}")
        print(f"  Source/Date: {md.get('source','')} | {md.get('date','')}")
        print(f"  Link: {md.get('link','')}")
        print("  Snippet:", doc[:200].replace("\n", " "), "...")