# Part 1, Question 6: holdout + retrieval evaluation

import os
from pathlib import Path

import chromadb
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ai_agents_jobs_2026"
EMBED_MODEL = "text-embedding-3-small"

print("DB_DIR:", DB_DIR)
print("DB_DIR exists:", DB_DIR.exists())

# Load API key (env var or openai.txt in project root)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = (PROJECT_ROOT / "openai.txt").read_text().strip()

oai = OpenAI(api_key=api_key)

client = chromadb.PersistentClient(path=str(DB_DIR))
print("Collections:", [c.name for c in client.list_collections()])

collection = client.get_collection(name=COLLECTION_NAME)
print("Using collection:", collection.name, "count:", collection.count())

test_queries = [
    "reinforcement learning agent roles in finance",
    "multi-agent orchestration using LangGraph",
    "tool-calling agents in production systems",
]

print("\nEmbedding queries...")
q_resp = oai.embeddings.create(model=EMBED_MODEL, input=test_queries)
q_vecs = [d.embedding for d in q_resp.data]
print("Query embeddings:", len(q_vecs), "dim:", len(q_vecs[0]))

print("\nQuerying Chroma...")
results = collection.query(
            query_embeddings=q_vecs,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

for qi, q in enumerate(test_queries):
    print("\n==============================")
    print(f"Query #{qi + 1}: {q}")

    for rank in range(3):
        md = results["metadatas"][qi][rank]
        doc = results["documents"][qi][rank]
        dist = results["distances"][qi][rank]
        print(f"\n  Rank {rank + 1} | distance={dist:.4f}")
        print(f"  {md.get('title', '')} | {md.get('source', '')} | {md.get('date', '')}")
        print("  Link:", md.get("link", ""))
        print("  Snippet:", doc[:200].replace("\n", " "), "...")