# Part 1, Question 3: chunking experiments

import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_PATH = BASE_DIR.parent / "data" / "ai_agents_jobs" / "AI_Agents_Ecosystem_2026.csv"

def row_to_doc(row) -> str:
    title = str(row.get("Title", "")).strip()
    source = str(row.get("Source", "")).strip()
    date = row.get("Date", "")
    date_str = "" if pd.isna(date) else str(date)

    desc = str(row.get("Description", "")).strip()
    return "\n".join([
        f"TITLE: {title}",
        f"SOURCE: {source}",
        f"DATE: {date_str}",
        f"DESCRIPTION: {desc}",
    ])

def chunk_docs(df, chunk_size=700, chunk_overlap=100, min_chunk_chars=40):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks, metadatas = [], []

    for row_id, row in df.iterrows():
        text = row.get("doc_text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        row_chunks = splitter.split_text(text)
        for chunk_id, chunk in enumerate(row_chunks):
            chunk = chunk.strip()
            if len(chunk) < min_chunk_chars:
                continue

            chunks.append(chunk)
            metadatas.append({
                "row_id": int(row_id),
                "chunk_id": int(chunk_id),
                "title": str(row.get("Title", ""))[:200],
                "source": str(row.get("Source", ""))[:60],
                "date": "" if pd.isna(row.get("Date", None)) else str(row.get("Date", ""))[:32],
                "link": str(row.get("Link", ""))[:500],
            })

    return chunks, metadatas

def main():
    df = pd.read_csv(DATA_PATH)

    # normalize Date early
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Description"] = df["Description"].fillna("").astype(str)

    print("Dataset shape:", df.shape)
    print(df.head(3))

    df["doc_text"] = df.apply(row_to_doc, axis=1)

    chunks, metadatas = chunk_docs(df, chunk_size=700, chunk_overlap=100)
    print("Rows:", len(df))
    print("Total chunks:", len(chunks))
    if chunks:
        print("Example chunk:\n", chunks[0])
        print("Example metadata:\n", metadatas[0])

    texts = df["doc_text"].dropna().astype(str).tolist()
    configs = [
        {"chunk_size": 350, "chunk_overlap": 50},
        {"chunk_size": 700, "chunk_overlap": 100},
        {"chunk_size": 1000, "chunk_overlap": 150},
    ]

    for cfg in configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        total_chunks = sum(len(splitter.split_text(t)) for t in texts)
        print(cfg, "=> avg chunks/row:", round(total_chunks / len(texts), 3), " total:", total_chunks)

if __name__ == "__main__":
    main()
