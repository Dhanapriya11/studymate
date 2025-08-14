from typing import List, Tuple
from .pdf_ingest import Chunk
from .llm_hf import HFLLM

class RAGPipeline:
    def __init__(self, vector_store, llm: HFLLM):
        self.vs = vector_store
        self.llm = llm

    def build_context(self, chunks: List[Chunk]) -> str:
        blocks = []
        for c in chunks:
            header = f"[Source: {c.pdf_name}, Page {c.page_num}, ID {c.chunk_id}]"
            blocks.append(f"{header}\n{c.text}")
        return "\n\n".join(blocks)

    def answer(self, question: str, top_k: int = 4) -> Tuple[str, List[Chunk]]:
        retrieved = self.vs.search(question, k=top_k)
        context = self.build_context(retrieved)
        system_prompt = (
            "You are StudyMate, an academic assistant. Answer ONLY using the provided context. "
            "Respond in bullet points (•) only — no paragraphs, introductions, or conclusions. "
            "Each bullet point must be a single relevant fact or idea from the context. "
            "Do not include information not found in the context. "
            "Cite the sources inline using [Page X, FileName]. "
            "If the answer is not in the context, say exactly: "
            "'I don't have enough information to answer that from the provided context.'"
        )
        prompt = f"""
{system_prompt}

<context>
{context}
</context>

Question: {question}
Answer with bullet points only, using exact relevant content from the context:
"""
        output = self.llm.generate(prompt)
        if "Error generating response" in output or "Unable to generate response" in output:
            error_context = (
                f"The LLM to generate a response. Here's the context that should have been used:\n\n{context}"
            )
            return error_context, retrieved
        return output, retrieved
