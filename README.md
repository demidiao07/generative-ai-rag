# Generative AI – Assignment 1 (Part 1)
**Name:** Wenshu Diao

## Overview
This project builds a vector database from the Kaggle dataset
*AI Agents Jobs Ecosystem 2026 – Real World* and evaluates semantic
retrieval using text embeddings.

## Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/nudratabbas/ai-agents-jobs-ecosystem-2026-real-world

Do NOT commit the dataset to the repository.

## Structure
- `notebooks/load_data.ipynb`: data loading, document construction, chunking
- `notebooks/text_embedding.ipynb`: embedding generation and retrieval tests
- `src/build_chroma.py`: vector database creation and upload

## Setup
```bash
pip install -r requirements.txt
