"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation dạng [ID] với ID của document trong context.
Nếu thiếu evidence, hãy từ chối xác minh."""
SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | ID: {chunk['id']} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if not api_key or not model:
            raise ValueError("Set OPENAI_API_KEY and OPENAI_MODEL in .env")
        client_args = {"api_key": api_key, "timeout": 30.0}
        if base_url:
            client_args["base_url"] = base_url
        response = OpenAI(**client_args).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content
    elif provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not api_key or not model:
            raise ValueError("Set GEMINI_API_KEY and LLM_MODEL in .env")
        response = genai.Client(api_key=api_key).models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        answer = response.text
    elif provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not api_key or not model:
            raise ValueError("Set ANTHROPIC_API_KEY and LLM_MODEL in .env")
        response = Anthropic(api_key=api_key, timeout=30.0).messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
        )
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("LLM returned an empty answer")
    return answer.strip()


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    refusal = {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    try:
        chunks = retrieve(query, top_k=top_k)
        if not chunks:
            return refusal
        context = format_context(reorder_for_llm(chunks))
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        return refusal

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0]["retrieval_method"],
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
