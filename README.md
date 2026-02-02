# Generative AI Assignment 1 – Part 1
**Name:** Wenshu (Demi) Diao

## Overview
This project implements a vector database pipeline for AI agent–related content using:
- document chunking
- text embeddings
- ChromaDB for vector storage
- similarity-based retrieval

## Project Structure

```markdown
src/
  chroma_smoketest.py
  chunk_smoketest.py
  embedding_smoketest.py
  ingest_to_chroma.py
  holdout_retrieval_test.py
  retrieval_examples.py
requirements.txt
sample_outputs.ipynb
README.md
.gitignore
```

## Setup Instructions
### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API key
Set the API key as an environment variable (I do not upload my API key):

```
export OPENAI_API_KEY="your_api_key_here"
```

## Dataset
The dataset **AI_Agents_Ecosystem_2026.csv** contains structured information about emerging AI agent roles, required skills, tools, and industry applications observed primarily in 2025–2026. The data focuses on modern agentic workflows, multi-agent orchestration frameworks (e.g., LangGraph), and applied use cases across finance, software engineering, and enterprise automation.

### Dataset Scope Filtering
To ensure the dataset is outside the base language model’s knowledge scope, the ingestion pipeline filters records to dates after October 2023. This ensures that the majority of the content reflects ecosystem developments that occurred after common LLM training cutoffs.

### Why the Dataset Is Out of Scope of the LLM
The language models and embedding models used in this project have a knowledge cutoff in late 2023. Many of the roles, tools, and workflows represented in this dataset emerged after that cutoff, meaning the base model cannot reliably answer questions about them without retrieval.

### Why RAG Is Appropriate 
Because the dataset contains recent, domain-specific information that evolves rapidly, retrieval-augmented generation is necessary to ground model responses in up-to-date facts and reduce hallucinations when answering questions about modern AI agent ecosystems.

**Dataset source (Kaggle):**  
👉 [AI Agents Jobs Ecosystem 2026 – Real World](https://www.kaggle.com/datasets/nudratabbas/ai-agents-jobs-ecosystem-2026-real-world)

**Note:** The dataset is **not included** in this repository in accordance with assignment instructions.

After downloading, place the CSV file at:
```text
data/ai_agents_jobs/AI_Agents_Ecosystem_2026.csv
```

## How to Run 

### Q1: ChromaDB smoketest

I chose ChromaDB as my vector database because it is free, open-source, and supports persistent local storage. I created a persistent client, created a collection, inserted sample documents with metadata, and verified similarity-based retrieval using a test query.

```bash
python src/chroma_smoketest.py
```

### Q3: Chunking smoketest + chunk size experiments

```bash
python src/chunk_smoketest.py
```

I tested chunking configurations:
- **350 / 50**: many small chunks (risk of fragmented context)
- **1000 / 150**: fewer large chunks (risk of retrieving irrelevant context)
- **700 / 100**: balanced retrieval specificity and contextual completeness
Based on these experiments, **700 / 100** was selected for ingestion.

### Q4: Embedding model smoketest (FastEmbed)

I tested the FastEmbed model `BAAI/bge-small-en-v1.5` by embedding semantically related and unrelated texts and computing cosine similarity. Related texts consistently exhibited higher similarity scores than unrelated texts, indicating that the embedding space captures semantic relationships effectively.

```bash
python src/embedding_smoketest.py
```

### Q5: Ingest chunk embeddings into ChromaDB

I chunked each document using LangChain’s `RecursiveCharacterTextSplitter` (700 characters with 100 overlap), embedded all chunks using OpenAI’s `text-embedding-3-small`, and stored vectors with associated metadata in a persistent ChromaDB collection.

```bash
python src/ingest_to_chroma.py
```

### Q6: Retrieval validation (OpenAI-embedded queries)

I validated retrieval quality by embedding representative queries using the same model (`text-embedding-3-small`, 1536 dimensions) and retrieving the nearest neighbors from ChromaDB. For each query, the top-ranked chunks were thematically aligned with the query topic (e.g., reinforcement learning roles, multi-agent orchestration, tool-calling agents), indicating that the vector index behaves as expected.

```bash
python src/holdout_retrieval_test.py
```

### Q7: Additional retrieval examples

I ran five representative natural-language queries and retrieved the top-5 most similar chunks from ChromaDB. Returned results included metadata (title, source, date, link) to support traceability and future citation in a RAG pipeline.

```bash
python src/retrieval_examples.py
```

## Sample Outputs
See `sample_outputs.ipynb` for:
- chunking experiments with different chunk sizes
- embedding dimension sanity checks
- ChromaDB retrieval examples with top-k neighbors
