import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
)


def show_sources(sources: list[dict]) -> None:
    """Show the source IDs used by citations and their retrieval details."""
    if not sources:
        return

    with st.expander(f"Nguồn tham khảo ({len(sources)})"):
        for source in sources:
            metadata = source["metadata"]
            st.markdown(f"**[{source['id']}] {metadata['title']}**")
            st.caption(
                f"Nguồn: {metadata['source']} · "
                f"Phương thức: {source['retrieval_method']} · "
                f"Điểm: {source['score']:.3f}"
            )
            if metadata.get("url"):
                st.link_button("Mở tài liệu gốc", metadata["url"])
            st.text(source["content"])
            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Hỏi đáp dựa trên tài liệu đã được lập chỉ mục.")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("RAG Chatbot")
st.caption("Đặt câu hỏi để nhận câu trả lời kèm nguồn tham khảo.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
            result = generate_with_citation(query, top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": result["retrieval_source"],
        }
    )
