from types import SimpleNamespace

import pytest

from src.contracts import validate_generation_result
from src import task10_generation as generation


def chunk(item_id: str, score: float = 0.8) -> dict:
    return {
        "id": item_id,
        "content": f"Evidence for {item_id}",
        "score": score,
        "metadata": {
            "source": "policy.md",
            "title": "Policy",
            "doc_type": "legal",
            "url": None,
            "chunk_index": 0,
        },
        "retrieval_method": "hybrid",
    }


def test_openai_uses_provider_specific_env_and_base_url(monkeypatch):
    import openai

    captured = {}

    def create(**kwargs):
        captured["request"] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  Answer [chunk-0]  "))]
        )

    def client(**kwargs):
        captured["client"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("LLM_MODEL", "different-model")
    monkeypatch.setattr(openai, "OpenAI", client)

    assert generation.call_llm("System", "Question") == "Answer [chunk-0]"
    assert captured["client"]["api_key"] == "test-key"
    assert captured["client"]["base_url"] == "https://example.test/v1"
    assert captured["request"]["model"] == "test-model"
    assert captured["request"]["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
    ]


def test_gemini_and_anthropic_dispatch(monkeypatch):
    from google import genai
    import anthropic

    monkeypatch.setenv("LLM_MODEL", "provider-model")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    def gemini_client(**kwargs):
        assert kwargs["api_key"] == "gemini-test"

        def generate_content(**request):
            assert request["model"] == "provider-model"
            assert request["contents"] == "Question"
            assert request["config"].system_instruction == "System"
            return SimpleNamespace(text="Gemini answer")

        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setattr(genai, "Client", gemini_client)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert generation.call_llm("System", "Question") == "Gemini answer"

    def anthropic_client(**kwargs):
        assert kwargs["api_key"] == "anthropic-test"

        def create(**request):
            assert request["model"] == "provider-model"
            assert request["system"] == "System"
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Anthropic answer")])

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(anthropic, "Anthropic", anthropic_client)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert generation.call_llm("System", "Question") == "Anthropic answer"


def test_generation_citations_refer_to_returned_source_ids(monkeypatch):
    chunks = [chunk("chunk-0", 0.9), chunk("chunk-1", 0.8), chunk("chunk-2", 0.7)]
    monkeypatch.setattr(generation, "retrieve", lambda query, top_k: chunks)

    def answer(system_prompt, user_message):
        assert "ID: chunk-0" in user_message
        assert "ID: chunk-1" in user_message
        assert "ID: chunk-2" in user_message
        assert user_message.index("ID: chunk-2") < user_message.index("ID: chunk-1")
        return "The policy applies [chunk-2]."

    monkeypatch.setattr(generation, "call_llm", answer)
    result = generation.generate_with_citation("What applies?", top_k=3)

    validate_generation_result(result)
    assert result["answer"] == "The policy applies [chunk-2]."
    assert result["sources"] == chunks
    assert result["retrieval_source"] == "hybrid"


@pytest.mark.parametrize("failure", ["empty", "provider", "retrieval"])
def test_generation_safe_refusal(monkeypatch, failure):
    if failure == "retrieval":
        def retrieve_error(query, top_k):
            raise RuntimeError("backend unavailable")

        monkeypatch.setattr(generation, "retrieve", retrieve_error)
    else:
        monkeypatch.setattr(
            generation, "retrieve", lambda query, top_k: [] if failure == "empty" else [chunk("chunk-0")]
        )

    if failure == "provider":
        def provider_error(system_prompt, user_message):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(generation, "call_llm", provider_error)

    result = generation.generate_with_citation("Question")
    validate_generation_result(result)
    assert result == {
        "answer": generation.SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }
