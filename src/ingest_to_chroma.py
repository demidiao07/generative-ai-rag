# Part 1, Question 5: embed + store + query

import os
from pathlib import Path

import numpy as np
import pandas as pd
import chromadb
from tqdm import tqdm
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = BASE_DIR.parent

# --- OpenAI key ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = (PROJECT_ROOT / "openai.txt").read_text().strip()
client = OpenAI(api_key=api_key)

# --- Load dataset ---
DATA_PATH = PROJECT_ROOT / "data" / "ai_agents_jobs" / "AI_Agents_Ecosystem_2026.csv"
df = pd.read_csv(DATA_PATH)
print("Raw dataset shape:", df.shape)

# Clean
df["Description"] = df["Description"].fillna("").astype(str)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df[df["Description"].str.len() >= 50].copy()
df = df.drop_duplicates(subset=["Link"]).reset_index(drop=True)

print("Clean dataset shape:", df.shape)

def row_to_doc(row) -> str:
    title = str(row.get("Title", "")).strip()
    source = str(row.get("Source", "")).strip()
    date = row.get("Date", None)
    date_str = "" if pd.isna(date) else date.strftime("%Y-%m-%d")
    desc = str(row.get("Description", "")).strip()

    return "\n".join([
        f"TITLE: {title}",
        f"SOURCE: {source}",
        f"DATE: {date_str}",
        f"DESCRIPTION: {desc}",
    ])

df["doc_text"] = df.apply(row_to_doc, axis=1)

# --- Chunking ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks, metadatas, ids = [], [], []

for row_id, row in df.iterrows():
    text = row["doc_text"]
    if not isinstance(text, str) or not text.strip():
        continue

    for chunk_id, chunk in enumerate(splitter.split_text(text)):
        chunk = chunk.strip()
        if len(chunk) < 40:
            continue

        ids.append(f"row{row_id}_chunk{chunk_id}")
        chunks.append(chunk)
        metadatas.append({
            "row_id": int(row_id),
            "chunk_id": int(chunk_id),
            "title": str(row.get("Title", ""))[:200],
            "source": str(row.get("Source", ""))[:60],
            "date": "" if pd.isna(row.get("Date", None)) else row["Date"].strftime("%Y-%m-%d"),
            "link": str(row.get("Link", ""))[:500],
        })

print(f"Total chunks created: {len(chunks)}")

# --- Embeddings ---
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 128

all_vectors = []
for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
    batch = chunks[start:start + BATCH_SIZE]
    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    all_vectors.extend([d.embedding for d in resp.data])

all_vectors = np.array(all_vectors, dtype=np.float32)
print("Embeddings shape:", all_vectors.shape)

# --- ChromaDB (rerun-safe rebuild) ---
DB_DIR = PROJECT_ROOT / "chroma_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

COLLECTION_NAME = "ai_agents_jobs_2026"
try:
    chroma_client.delete_collection(name=COLLECTION_NAME)
except Exception:
    pass

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

UPSERT_BATCH = 200
for start in tqdm(range(0, len(chunks), UPSERT_BATCH), desc="Writing to Chroma"):
    end = start + UPSERT_BATCH
    collection.add(
        ids=ids[start:end],
        documents=chunks[start:end],
        metadatas=metadatas[start:end],
        embeddings=all_vectors[start:end].tolist(),
    )

print("Collection count:", collection.count())

# --- Query sanity check ---
query = "reinforcement learning agent roles"
q_vec = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding

results = collection.query(
    query_embeddings=[q_vec],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

for i, md in enumerate(results["metadatas"][0]):
    doc = results["documents"][0][i]
    dist = results["distances"][0][i]
    print(f"\nRank {i+1} | distance={dist:.4f}")
    print("Title:", md.get("title"))
    print("Source:", md.get("source"))
    print("Date:", md.get("date"))
    print("Link:", md.get("link"))
    print("Snippet:", doc[:250], "...")