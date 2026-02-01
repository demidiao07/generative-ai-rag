# generative-ai-rag

For this project, I use the AI Agents Jobs Ecosystem (2026) dataset, which contains job postings and descriptions related to emerging AI agent roles. Because this dataset reflects information from 2026, it is outside the knowledge cutoff of the LLM used in this project (October 2023), making retrieval-augmented generation necessary for accurate responses. The dataset contains rich natural language text and can be reused in later assignments for tasks such as classification, summarization, and fine-tuning.

**Question 2: Dataset and LLM Selection**
For this project, we use the AI Agents Jobs Ecosystem (2026) dataset. This dataset contains natural-language job titles, descriptions, and role requirements related to emerging AI agent–focused positions.

We choose this dataset for several reasons:
**1. Out of LLM knowledge scope**
The dataset reflects job market information from 2026, while the selected LLM (GPT-4o-mini) has a knowledge cutoff of October 2023. As a result, the base LLM cannot reliably answer questions about these roles without retrieval, making it well-suited for evaluating retrieval-augmented generation (RAG).

**2. Text-rich and suitable for embeddings**
Job descriptions contain unstructured natural language, which is ideal for semantic embeddings, similarity search, and retrieval tasks.

**3. Reusable for future assignments**
The dataset can be reused in later assignments for few-shot prompting and fine-tuning tasks, such as role classification, skill extraction, or summarization.

**4. Practical size and accessibility**
Compared to very large social-media datasets, this dataset is easier to download and process while still being sufficiently complex to demonstrate RAG techniques.
We use GPT-4o-mini as the LLM in this project because it supports in-context learning and reasoning while remaining computationally efficient.
