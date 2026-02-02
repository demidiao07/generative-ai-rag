import os
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# -----------------------
# Config
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "ai_agents_jobs_2026"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

TOP_K = 5
CANDIDATES_K = 40


# -----------------------
# OpenAI key
# -----------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = (PROJECT_ROOT / "openai.txt").read_text().strip()
os.environ["OPENAI_API_KEY"] = api_key


# -----------------------
# Vector store
# -----------------------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)
print("Chroma count:", vectorstore._collection.count())


# -----------------------
# Helpers: filtering (optional)
# -----------------------
def is_arxiv(doc) -> bool:
    md = doc.metadata or {}
    src = (md.get("source", "") or "").lower()
    title = (md.get("title", "") or "").lower()
    link = (md.get("link", "") or "").lower()
    text = (doc.page_content or "").lower()
    return ("arxiv" in src) or ("arxiv" in title) or ("arxiv" in link) or ("arxiv" in text)

def retrieve_candidates(query: str) -> List:
    docs = vectorstore.similarity_search(query, k=CANDIDATES_K)
    # keep arxiv too unless you really want jobs-only; your dataset is mixed, so mixed is fine
    return docs

def format_docs(docs: List) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        header = f"[{i}] title={md.get('title','')} | source={md.get('source','')} | date={md.get('date','')} | link={md.get('link','')}"
        parts.append(header + "\n" + (d.page_content or ""))
    return "\n\n".join(parts)


# -----------------------
# LLM + prompts
# -----------------------
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
parser = StrOutputParser()

base_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's question as best you can."),
    ("user", "{question}")
])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Use the provided context to answer. If the context is only partially relevant, "
     "answer only what is supported and explicitly say what is missing. Do not invent facts."),
    ("user",
     "Question: {question}\n\nContext:\n{context}\n\nAnswer (cite chunk numbers like [1], [2]):")
])

hyde_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Write a short hypothetical answer that would likely appear in the target documents. "
     "Do not cite sources. The goal is to help retrieval."),
    ("user", "Question: {question}\nHypothetical answer:")
])

rerank_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are reranking retrieved passages for relevance to the question. "
     "Return ONLY a comma-separated list of the best passage numbers, in order, like: 3,1,5,2,4."),
    ("user",
     "Question: {question}\n\nPassages:\n{context}\n\nBest {top_k} passage numbers:")
])


# -----------------------
# 1) Original LLM (no RAG)
# -----------------------
def answer_no_rag(question: str) -> str:
    return (base_prompt | llm | parser).invoke({"question": question})


# -----------------------
# 2) Simple RAG
# -----------------------
def answer_simple_rag(question: str) -> str:
    docs = retrieve_candidates(question)[:TOP_K]
    context = format_docs(docs)
    return (rag_prompt | llm | parser).invoke({"question": question, "context": context})


# -----------------------
# 3) RAG + HyDE (query rewriting)
# -----------------------
def answer_hyde_rag(question: str) -> str:
    hypothetical = (hyde_prompt | llm | parser).invoke({"question": question})
    # retrieve using HyDE text (often better than raw question)
    docs = retrieve_candidates(hypothetical)[:TOP_K]
    context = format_docs(docs)
    return (rag_prompt | llm | parser).invoke({"question": question, "context": context})


# -----------------------
# 4) RAG + reranking (LLM rerank)
# -----------------------
def answer_rerank_rag(question: str) -> str:
    candidates = retrieve_candidates(question)  # many docs
    context_all = format_docs(candidates)

    order = (rerank_prompt | llm | parser).invoke({
        "question": question,
        "context": context_all,
        "top_k": TOP_K
    })

    # parse "3,1,5,2,4"
    try:
        idxs = [int(x.strip()) for x in order.split(",") if x.strip().isdigit()]
        idxs = [i for i in idxs if 1 <= i <= len(candidates)]
    except Exception:
        idxs = list(range(1, min(TOP_K, len(candidates)) + 1))

    chosen = [candidates[i - 1] for i in idxs[:TOP_K]]
    context = format_docs(chosen)
    return (rag_prompt | llm | parser).invoke({"question": question, "context": context})


# -----------------------
# Run evaluation on 5 questions
# -----------------------
if __name__ == "__main__":
    questions = [
        "What tools/frameworks are mentioned for multi-agent orchestration in this dataset? Give examples.",
        "Across the dataset, what are common job requirements for AI Agent Engineer roles?",
        "Find sources discussing tool-integrated reasoning. What is the core idea and what tasks does it target?",
        "What themes appear about agent evaluation, reliability, or monitoring in 2026 sources?",
        "Which sources mention LoRA fine-tuning and what are they using it for?",
    ]

    for i, q in enumerate(questions, start=1):
        print("\n" + "=" * 90)
        print(f"Q{i}: {q}")

        print("\n--- Original LLM (no RAG) ---")
        print(answer_no_rag(q))

        print("\n--- Simple RAG ---")
        print(answer_simple_rag(q))

        print("\n--- RAG + HyDE ---")
        print(answer_hyde_rag(q))

        print("\n--- RAG + Reranking ---")
        print(answer_rerank_rag(q))