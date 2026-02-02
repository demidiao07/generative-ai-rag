# Part 2, Question 2

# pip install -U langchain langchain-openai langchain-chroma chromadb

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------
# Paths & constants
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "ai_agents_jobs_2026"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5


# -----------------------
# Load OpenAI API key (env var or openai.txt)
# -----------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = (PROJECT_ROOT / "openai.txt").read_text().strip()
os.environ["OPENAI_API_KEY"] = api_key


# -----------------------
# Vector store (load persisted Chroma collection)
# -----------------------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)

print("Persist dir:", DB_DIR)
print("Collection name:", COLLECTION_NAME)
print("LangChain collection count:", vectorstore._collection.count())

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
    )


# -----------------------
# LLM
# -----------------------
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)


# -----------------------
# Prompt + basic RAG chain
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Use the provided context to answer. If the context is insufficient, say you don't know."),
    ("user",
     "Question: {question}\n\n"
     "Context:\n{context}\n\n"
     "Answer (cite chunk numbers like [1], [2] when relevant):")
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    q = "What are the main themes about AI agents in this dataset?"
    print(rag_chain.invoke(q))
