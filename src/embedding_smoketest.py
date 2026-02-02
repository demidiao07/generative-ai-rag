# Part 1, Question 4: embedding model + similarity

# pip install -U fastembed numpy
import numpy as np
from fastembed import TextEmbedding

texts = [
    "This role requires reinforcement learning and Python.",
    "Customer support specialist handling B2B client issues and churn prevention.",
    "LangGraph multi-agent orchestration with tool calling."
]

model_name = "BAAI/bge-small-en-v1.5"
embedder = TextEmbedding(model_name=model_name)

embeddings = np.array(list(embedder.embed(texts)), dtype=np.float32)

print("Model:", model_name)
print("Embedding shape:", embeddings.shape)

def cos_sim(a, b, eps=1e-12):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)

names = ["rl_python", "support", "langgraph_agents"]
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print(f"sim({names[i]}, {names[j]}) = {cos_sim(embeddings[i], embeddings[j]):.3f}")