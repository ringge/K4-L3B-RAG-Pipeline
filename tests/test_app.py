from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import task10_generation as generation


def test_chat_displays_and_preserves_cited_sources(monkeypatch):
    calls = []
    source = {
        "id": "chunk-0",
        "content": "Policy evidence",
        "score": 0.8,
        "retrieval_method": "hybrid",
        "metadata": {
            "title": "Policy",
            "source": "policy.md",
            "url": "https://example.test/policy",
        },
    }

    def generate(query, top_k):
        calls.append((query, top_k))
        return {
            "answer": "The policy applies [chunk-0].",
            "sources": [source],
            "retrieval_source": "hybrid",
        }

    monkeypatch.setattr(generation, "generate_with_citation", generate)
    app = AppTest.from_file(Path(__file__).parent.parent / "app.py").run()
    assert not app.exception

    app.chat_input[0].set_value("What applies?").run()
    assert not app.exception
    assert calls == [("What applies?", 5)]
    assert app.session_state["messages"][1]["sources"] == [source]
    assert app.session_state["messages"][1]["retrieval_source"] == "hybrid"
    assert any("[chunk-0] Policy" in item.value for item in app.markdown)
    assert any("Điểm: 0.800" in item.value for item in app.caption)

    app.run()
    assert not app.exception
    assert calls == [("What applies?", 5)]
    assert len(app.get("expander")) == 1
    assert any("[chunk-0] Policy" in item.value for item in app.markdown)
