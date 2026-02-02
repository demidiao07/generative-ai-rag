# Generative AI Assignment 1 – Part 1
**Name:** Demi (Wenshu Diao)

## Project Overview
This project builds a local vector database using ChromaDB, chunks and embeds a recent dataset, ingests embeddings into the DB, and validates retrieval quality via similarity search.

## Setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

I chose ChromaDB as my vector database because it is free, open-source, and supports persistent local storage. I created a persistent client, created a collection for my dataset, inserted sample documents with metadata, and verified retrieval by running a query. This collection will store embedded chunks of my dataset for the RAG pipeline.

This dataset contains articles and posts related to the AI agent ecosystem through January 2026. Since GPT-4o-mini has a knowledge cutoff in October 2023, all documents dated after October 1, 2023 are outside the model’s training knowledge. Therefore, retrieval-augmented generation (RAG) is necessary to answer questions about recent AI agent frameworks, tools, and trends.

## Dataset
The dataset AI_Agents_Ecosystem_2026.csv contains structured information about emerging AI agent roles, required skills, tools, and industry applications observed in 2025–2026. The data focuses on modern agentic workflows, multi-agent orchestration frameworks, and applied use cases across finance, software engineering, and enterprise automation.

Why the Dataset Is Out of Scope of the LLM The LLM used in this project (e.g., GPT-4o-mini) has a knowledge cutoff in late 2023. Many of the roles, tools, and ecosystem developments represented in this dataset emerged after that cutoff, making the information unavailable to the base model without retrieval.

Why RAG Is Appropriate Because the dataset contains recent, domain-specific information that evolves rapidly, retrieval-augmented generation is necessary to ground model responses in up-to-date facts and reduce hallucinations when answering questions about modern AI agent ecosystems.

Download from Kaggle:
https://www.kaggle.com/datasets/nudratabbas/ai-agents-jobs-ecosystem-2026-real-world

Download the dataset and place it here:
data/ai_agents_jobs/AI_Agents_Ecosystem_2026.csv

## How to Run (Part 1)

### Q1: ChromaDB smoketest
```bash
python src/chroma_smoketest.py
```

### Q3: Chunking smoketest + chunk size experiments
```bash
python src/chunk_smoketest.py
```

### Q4: Embedding model smoketest (FastEmbed)
```bash
python src/embedding_smoketest.py
```

### Q5: Ingest chunk embeddings into ChromaDB
```bash
python src/ingest_to_chroma.py
```

### Q6: Retrieval validation (OpenAI-embedded queries)
```bash
python src/holdout_retrieval_test.py
```

### Q7: Additional retrieval examples (optional)
```bash
python src/retrieval_examples.py
```

Question 6:
I validated retrieval quality by querying the ChromaDB collection using embeddings from the same model used for ingestion (text-embedding-3-small, 1536 dimensions). For each test query, the top retrieved chunks were thematically aligned with the query topic (e.g., reinforcement learning papers for RL queries, orchestration results for multi-agent orchestration, and tool-safety/function-calling results for tool-calling queries). This indicates that the embedding model + vector database return nearest neighbors that are meaningful in the original text space.

Question 7:I queried the persisted ChromaDB collection using multiple natural-language queries embedded with the same model used during ingestion (text-embedding-3-small). For each query, the top-k retrieved chunks were semantically aligned with the query topic, demonstrating consistent and meaningful nearest-neighbor retrieval behavior in the vector database.

