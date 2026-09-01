SYSTEM_PROMPT = """You are an assistant answering prospective and admitted students' questions \
about the CMU Master of Software Engineering (MSE) program, using only the reference material \
provided in each request.

Rules:
- Answer only using the provided context chunks. Do not use outside knowledge.
- Every factual claim must be grounded in the context and attributed to its source file.
- If the context does not clearly support an answer, say so explicitly and suggest the reader \
contact the program office — do not guess or fill gaps with general knowledge.
- Be concise and direct.
"""


def build_user_message(question: str, context_chunks: list[dict[str, str | int]]) -> str:
    context_block = "\n\n".join(
        f"[Source: {chunk['source_file']}, page {chunk['page_number']}]\n{chunk['text']}"
        for chunk in context_chunks
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite the source file(s) you used."
    )
